"""基于 OpenAI 官方 SDK 与 SGLang 驱动的 TuRBO LLM 元优化控制器"""

import json
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    import urllib.error
    import urllib.request
    HAS_OPENAI = False

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True

    class MetaActionModel(BaseModel):
        """TuRBO 元控制动作的 Pydantic 结构化校验模型"""

        thought: str = Field(
            default="Default heuristic reasoning.",
            description="针对当前气动收敛状态与贝叶斯优化探索/利用平衡的工程推理",
        )
        length_scale_factor: float = Field(
            default=1.0,
            ge=0.5,
            le=2.0,
            description="信任域长度 L 的动态缩放系数 (范围 [0.5, 2.0])",
        )
        prob_perturb: float = Field(
            default=0.5,
            ge=0.1,
            le=1.0,
            description="18 维 Kulfan 几何参数扰动概率 (范围 [0.1, 1.0])",
        )
        exploration_weight: float = Field(
            default=1.0,
            ge=0.5,
            le=2.0,
            description="Thompson Sampling 后验不确定性探索缩放 (范围 [0.5, 2.0])",
        )
        suggested_point: Optional[Sequence[float]] = Field(
            default=None,
            description="当判断优化陷入停滞时，可直接指定下一步采样点的 18 维归一化坐标 [0, 1]^18 (长度为 18 的浮点数列表)；若无需直接指定，设为 null",
        )
except ImportError:
    HAS_PYDANTIC = False


SYSTEM_PROMPT = """You are an expert aerodynamicist and Bayesian Optimization meta-controller.
Your task is to dynamically adjust the hyperparameters of a Trust Region Bayesian Optimization (TuRBO-1) algorithm that is optimizing an airfoil shape for maximum Lift-to-Drag ratio (L/D) under transonic/subsonic flow.

Every batch of CFD evaluations, you receive the current optimization state:
1. Physical conditions (Airfoil, Reynolds number, Alpha angle, CFD budget).
2. Optimization progress & performance (initial L/D, best L/D, recent window improvements, CL, CD, max thickness).
3. Current best airfoil geometry (18-dimensional normalized coordinates in [0, 1]^18: upper weights 0-7, lower weights 8-15, leading edge weight 16, trailing edge thickness 17).
4. TuRBO internal dynamics (trust region length L, consecutive successes/failures, stagnation count).

CRITICAL INSTRUCTIONS FOR REASONING EFFICIENCY:
- Keep your thinking process EXTREMELY BRIEF (strictly under 60 words).
- DO NOT perform manual arithmetic, math derivations, or coordinate-by-coordinate calculations in your thinking.
- Formulate a 1-2 sentence high-level aerodynamic rationale, then IMMEDIATELY finish thinking and output the final JSON object.

JSON fields required:
- "thought": A brief 1-2 sentence engineering rationale.
- "length_scale_factor": float in [0.5, 2.0]. (Multiplier for trust region length L. Expand >1.0 if consistent progress in a gradient direction; shrink <1.0 if oscillating or near convergence).
- "prob_perturb": float in [0.1, 1.0]. (Probability of mutating each of the 18 Kulfan shape weights. Lower values like 0.2~0.4 focus on fine-tuning local shape coordinates; higher values like 0.8~1.0 perform global exploration).
- "exploration_weight": float in [0.5, 2.0]. (Multiplier for Thompson Sampling posterior uncertainty. Higher boosts exploration, lower favors exploitation).
- "suggested_point": null OR a list of exactly 18 floats in [0.0, 1.0].
  * If the optimizer is progressing normally, set this to null to let TuRBO's Thompson Sampling select the candidate.
  * If the optimizer has stagnated or is trapped in an inferior local region, you can directly override TuRBO's next candidate: take the Current Best Normalized Parameters provided above and apply small targeted perturbations (+/- 0.02 to 0.08) to key parameters (such as upper camber/thickness or LE radius) directly in the final JSON array.

Respond ONLY with the final valid JSON object.
"""


class LLMMetaAgent:
    """使用 OpenAI 客户端管理连接池、结构化输出与 Qwen 官方推荐采样配置的元控制器"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30000/v1",
        model: str = "Qwen/Qwen3.8-27B-FP8",
        api_key: str = "EMPTY",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.endpoint = f"{self.base_url}/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

        if HAS_OPENAI:
            # 禁用自动多重重试 (max_retries=0)，避免中途断开导致 SGLang 丢弃正在生成的会话并从头 Prefill
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=0,
            )
        else:
            self.client = None

    def build_user_prompt(self, state: Dict[str, Any]) -> str:
        """构建结构化状态提示词"""
        best_pt = state.get("best_point_unit")
        geom_str = ""
        if best_pt and len(best_pt) == 18:
            u_w = [round(float(x), 4) for x in best_pt[0:8]]
            l_w = [round(float(x), 4) for x in best_pt[8:16]]
            le_w = round(float(best_pt[16]), 4)
            te_t = round(float(best_pt[17]), 4)
            geom_str = (
                f"\n=== Current Best Normalized Parameters ([0, 1]^18) ===\n"
                f"- Upper Weights (0-7): {u_w}\n"
                f"- Lower Weights (8-15): {l_w}\n"
                f"- LE Weight (16): {le_w} | TE Thickness (17): {te_t}\n"
            )

        prompt = f"""=== Current Optimization State ===
- Airfoil: {state.get('airfoil_name', 'N/A')} | Re: {state.get('reynolds', 0):.2e} | Alpha: {state.get('alpha', 0)} deg
- Progress: Step {state.get('step', 0)} / {state.get('max_evals', 500)} (Remaining Budget: {state.get('budget_left', 0)})
- Initial L/D: {state.get('init_ld', 0):.2f} | Global Best L/D: {state.get('best_ld', 0):.2f} (Total Gain: {state.get('total_gain', 0):.2f}%)
- Current Best Aerodynamics: CL = {state.get('best_cl', 0):.4f}, CD = {state.get('best_cd', 0):.5f}, t_max = {state.get('best_t_max', 0)*100:.2f}% (Valid Range: 8% ~ 22%)
{geom_str}
=== Recent Window ===
- Window Best L/D: {state.get('window_best_ld', 0):.2f} | Window Mean L/D: {state.get('window_mean_ld', 0):.2f}
- Improvements in Last Window: {state.get('window_improvements', 0)}
- Consecutive Stagnation Steps: {state.get('stagnation_count', 0)}

=== TuRBO Trust Region State ===
- Current TR Length L: {state.get('tr_length', 0.8):.4f} (Valid Range: [{state.get('tr_min', 0.0078):.4f}, {state.get('tr_max', 1.6):.2f}])
- Success Counter: {state.get('success_counter', 0)} / {state.get('success_tolerance', 3)}
- Failure Counter: {state.get('failure_counter', 0)} / {state.get('failure_tolerance', 18)}
- Restarts So Far: {state.get('restart_count', 0)}

Concisely analyze in <60 words and output final JSON:"""
        return prompt

    def decide_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """向 SGLang 发送请求获取元动作，采用 Qwen 官方推荐 Thinking Mode 采样参数"""
        user_prompt = self.build_user_prompt(state)

        # 默认安全动作 (Fallback)
        default_action = {
            "thought": "Fallback to default TuRBO dynamics.",
            "length_scale_factor": 1.0,
            "prob_perturb": float(min(20.0 / state.get("dim", 18), 1.0)),
            "exploration_weight": 1.0,
            "suggested_point": None,
        }

        try:
            if HAS_OPENAI and self.client is not None:
                # 采用 Qwen 官方推荐的 Thinking Mode 采样超参，max_tokens 调宽至 3072 防止截断
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=1.0,
                    top_p=0.95,
                    presence_penalty=0.0,
                    extra_body={
                        "top_k": 20,
                        "min_p": 0.0,
                        "repetition_penalty": 1.0,
                    },
                    max_tokens=3072,
                )

                choice = response.choices[0]
                raw_text = choice.message.content or ""
                reasoning = getattr(choice.message, "reasoning_content", None)
                finish_reason = getattr(choice, "finish_reason", "")

                action = self._parse_json_response(raw_text, default_action)

                # 如果因为 token 耗尽被截断且未解析出 JSON，打印友好预警，不输出超长思维草稿
                if finish_reason == "length" and action.get("thought") == default_action["thought"]:
                    action["thought"] = "[LLM Warning] Thinking tokens exceeded limit before JSON completion. Falling back to default TuRBO dynamics."
                elif reasoning and action.get("thought") == default_action["thought"]:
                    # 若有简短 reasoning 且只占前两句，取前 150 字符
                    action["thought"] = reasoning.strip()[:150] + ("..." if len(reasoning.strip()) > 150 else "")

                return action
            else:
                # 若未安装 openai 库时的标准库回退
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "presence_penalty": 0.0,
                    "top_k": 20,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                    "max_tokens": 3072,
                }
                req_data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    self.endpoint,
                    data=req_data,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    raw_text = res_json["choices"][0]["message"]["content"].strip()
                    return self._parse_json_response(raw_text, default_action)

        except Exception as e:
            raise RuntimeError(
                f"[LLM Fatal Error in Meta-CFD] Live request to {self.base_url} failed with {type(e).__name__}: {e}. "
                "Silent degradation is strictly forbidden under research integrity rules."
            ) from e

    def _parse_json_response(
        self, text: str, default_action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """稳健解析并校验 LLM 输出"""
        clean_text = re.sub(r"```json\s*", "", text)
        clean_text = re.sub(r"```\s*", "", clean_text).strip()

        data = None
        try:
            data = json.loads(clean_text)
        except Exception:
            match = re.search(r"\{.*?\}", clean_text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except Exception:
                    pass

        if not isinstance(data, dict):
            return default_action

        # 提取 candidate direct point (兼容 suggested_point 与 direct_point)
        suggested_raw = data.get("suggested_point")
        if suggested_raw is None:
            suggested_raw = data.get("direct_point")

        suggested_cleaned = None
        if isinstance(suggested_raw, (list, tuple)) and len(suggested_raw) > 0:
            cleaned_list = []
            for v in suggested_raw:
                try:
                    val = float(v)
                    cleaned_list.append(max(0.0, min(1.0, val)))
                except (TypeError, ValueError):
                    cleaned_list.append(None)
            if any(x is not None for x in cleaned_list):
                suggested_cleaned = cleaned_list

        if HAS_PYDANTIC:
            try:
                data_copy = dict(data)
                data_copy["suggested_point"] = suggested_cleaned
                validated = MetaActionModel(**data_copy)
                res = validated.model_dump()
                res["suggested_point"] = suggested_cleaned
                return res
            except Exception:
                pass

        # 宽容模式手动截断
        return {
            "thought": str(data.get("thought", default_action["thought"])),
            "length_scale_factor": float(
                max(0.5, min(2.0, data.get("length_scale_factor", 1.0)))
            ),
            "prob_perturb": float(
                max(0.1, min(1.0, data.get("prob_perturb", 0.5)))
            ),
            "exploration_weight": float(
                max(0.5, min(2.0, data.get("exploration_weight", 1.0)))
            ),
            "suggested_point": suggested_cleaned,
        }
