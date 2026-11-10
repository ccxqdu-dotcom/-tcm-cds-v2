import os, json, urllib.request, urllib.error, re
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-sol")
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

SYSTEM_PROMPT = """
你是一个面向中医临床学习者的混合式临床决策支持推理层。目标是做结构化鉴别，不替代医生诊断，不提供个体化剂量。
严格按以下优先级推理：
1) 红旗与高风险背景；
2) 现代医学病因锚点/治疗相关不良反应/结构性疾病；
3) 既往治疗反应与哪些症状随治疗变化、哪些不变；
4) 把可能独立的症状链拆开，不强行用一个证解释全部；
5) 中医寒热虚实、气血痰湿水、升降出入；
6) 八轴：炎症、动力、屏障、微循环、神经自主、水液、代谢排泄、异常增殖；
7) 最后才在候选方剂中排序。

重要规则：
- 病因锚点优先于表面方证。
- 治疗反应不同步是强反证。
- 鼻腔脓性分泌物不能简单等同肺痰；囊肿/息肉/肿瘤不能简单等同痰湿/癥瘕。
- 儿童、肿瘤、高龄、多病共存、妊娠、抗凝、心律失常风险等必须降低处方确定性。
- 前列腺癌内分泌/ADT治疗 + 潮热多汗：优先考虑治疗相关下丘脑体温调节异常；中医方剂只作为辅助思路。
- 温胆汤等改善苔腻/腹胀而舌灼感不变：提示舌部主诉应换轴，考虑黏膜/感觉神经和继发性灼口原因。
- 鼻炎+眼痒+清嗓/咳：先考虑过敏性鼻炎/鼻后滴流链。
- 大便不干但排不出/排不净：考虑推进或出口障碍，不要机械归为津亏燥结。
- 对候选方必须写为什么选和反证/为什么不选。
- 若一个病例包含两个相对独立问题，明确输出 split_problem=true，并分别给两条路径。
- 不输出具体克数/剂量。
- 将动物、细胞、网络药理与人体临床证据明确区分。
- 对胸痛/胸闷心慌、呼吸困难、出血、进行性神经症状等给出必要的现代医学优先评估。
- 输出必须是纯 JSON，不要 markdown。
"""

def schema_example():
    return {
      "summary":"一句话概括",
      "split_problem":False,
      "problem_chains":[{"name":"主链","logic":"因果链说明"}],
      "disease_anchor":"最重要的现代病因锚点",
      "red_flags":["红旗/高风险"],
      "modern_differential":[{"item":"鉴别","why":"理由","priority":"high|medium|low"}],
      "tcm_pattern":{"core":"核心病机","secondary":["次级病机"],"negative_evidence":["反证"]},
      "axes":[{"name":"神经自主","score":3,"why":"理由"}],
      "formula_candidates":[
        {"name":"方剂名或某方思路","role":"主轴|辅助|备选","why":"为什么选","against":"反证/为什么不选","evidence":"A-E或不确定"}
      ],
      "tests":["建议检查"],
      "follow_up":["复评指标"],
      "questions":["最能改变排序的追问"],
      "confidence":"high|medium|low",
      "safety_note":"安全说明"
    }

def call_openai(payload):
    if not API_KEY:
        return {"ok":False,"error":"OPENAI_API_KEY_NOT_CONFIGURED"}
    user_prompt = {
      "case_text": payload.get("text",""),
      "local_rule_context": payload.get("local",{}),
      "available_formula_library": payload.get("formula_library",[]),
      "required_output_shape": schema_example()
    }
    body = {
      "model": MODEL,
      "reasoning": {"effort":"medium"},
      "instructions": SYSTEM_PROMPT,
      "input": json.dumps(user_prompt, ensure_ascii=False),
      "max_output_tokens": 5000
    }
    req = urllib.request.Request(
      "https://api.openai.com/v1/responses",
      data=json.dumps(body).encode("utf-8"),
      headers={"Authorization": "Bearer " + API_KEY, "Content-Type":"application/json"},
      method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data=json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail=e.read().decode("utf-8","ignore")
        return {"ok":False,"error":"OPENAI_HTTP_ERROR","detail":detail[:1200]}
    except Exception as e:
        return {"ok":False,"error":"OPENAI_REQUEST_FAILED","detail":str(e)}
    text=data.get("output_text","")
    if not text:
        chunks=[]
        for item in data.get("output",[]):
            if isinstance(item,dict):
                for c in item.get("content",[]):
                    if isinstance(c,dict) and c.get("type")=="output_text":
                        chunks.append(c.get("text",""))
        text="".join(chunks)
    text=text.strip()
    if text.startswith("~~~json"):
        text=text[7:]
    if text.startswith("~~~"):
        text=text[3:]
    if text.endswith("~~~"):
        text=text[:-3]
    text=text.strip()
    try:
        return {"ok":True,"model":MODEL,"result":json.loads(text)}
    except Exception:
        return {"ok":False,"error":"AI_JSON_PARSE_FAILED","raw":text[:6000]}

class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        base = super().translate_path(path)
        rel = os.path.relpath(base, os.getcwd())
        return str(ROOT / rel)

    def do_POST(self):
        if self.path != "/api/analyze":
            self.send_error(404); return
        try:
            length=int(self.headers.get("Content-Length","0"))
            payload=json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            self._json(400,{"ok":False,"error":"BAD_JSON"}); return
        if not payload.get("text","").strip():
            self._json(400,{"ok":False,"error":"EMPTY_CASE"}); return
        result=call_openai(payload)
        self._json(200 if result.get("ok") else 503,result)

    def do_GET(self):
        if self.path == "/api/status":
            self._json(200,{"ok":True,"ai_configured":bool(API_KEY),"model":MODEL}); return
        return super().do_GET()

    def _json(self,status,obj):
        data=json.dumps(obj,ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Cache-Control","no-store")
        self.send_header("Content-Length",str(len(data)))
        self.end_headers()
        self.wfile.write(data)

if __name__=="__main__":
    os.chdir(ROOT)
    port=int(os.getenv("PORT","10000"))
    print("Hybrid clinical decision app serving on", port, "AI configured=", bool(API_KEY), "model=", MODEL)
    ThreadingHTTPServer(("0.0.0.0",port),Handler).serve_forever()
