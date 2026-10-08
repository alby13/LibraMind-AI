"""A calculator tool for LibraMind: a safe arithmetic evaluator and the loop that lets the model call it.

LibraMind can't do arithmetic reliably in its head (2-digit addition 4%, see matheval.py), but it was trained to
call functions. Here it gets one function, calculate(expression). When a reply contains a call, the expression is
evaluated exactly and the result goes back in a <tool_response>, in the same format as the function-calling
training data; then the model continues and writes its answer. This repeats for several calculations.

The calculator parses expressions with Python's ast module and evaluates only numbers, + - * / // % **, parentheses
and a few math functions; nothing is ever executed as code.
"""
import ast
import json
import math
import operator as op
import re

# no example expression in the description: the model copied one into unrelated chats
TOOL = {"type": "function", "function": {
    "name": "calculate",
    "description": "Evaluate an arithmetic expression exactly and return the result. Use it for every calculation.",
    "parameters": {"type": "object", "properties": {"expression": {
        "type": "string", "description": "Numbers, the operators + - * / ( ) % and sqrt() only, built from the numbers in the question",
        # commas only as thousands separators: a free comma let forced calls degenerate into "25.00,20%20%..."
        "pattern": r"^(?:[0-9+\-*/().%x×÷^ ]|,[0-9]{3}|of|sqrt){1,60}$"}},
        "required": ["expression"]}}}
TOOLS_PROMPT = ("You can call the following functions:\n<tools>\n{tools}\n</tools>\n\nTo call a function, reply with a "
                "JSON object inside <tool_call></tool_call> tags, for example:\n<tool_call>\n"
                '{{"name": "function_name", "arguments": {{"arg": "value"}}}}\n</tool_call>\n'
                "You may make several calls. Function results come back in <tool_response> tags.")
CALC_INSTRUCTIONS = ("You are LibraMind, a friendly AI made by alby13. You are bad at mental arithmetic, so for any "
                     "calculation, call the calculate function instead of working it out yourself. Break word problems "
                     "into small steps and make one call per step; after the last result, write the answer.")
NUMBER_WORDS = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|"
                          r"forty|fifty|hundred|thousand|million|half|twice|double|triple|dozen)\b", re.I)
MATH_WORDS = re.compile(r"\b(how (many|much)|total|sum|plus|minus|times|multipl\w*|divid\w*|percent\w*|average|"
                        r"cost|costs|price|discount|off|sale|each|per|left|remain\w*|calculat\w*|add|subtract|"
                        r"square root|squared|twice as|half of)\b", re.I)
ARITHMETIC = re.compile(r"[\d)]\s*[-+*/×÷x^%]\s*[\d(]|\d\s*%|\bsqrt\b")
# a question that IS an expression ("45 + 76", "12% of 50", "sqrt(2) * 10") goes to the calculator tool; word
# problems (even ones with a percentage in them) do better reasoning in text with the inline calculator
EXPLICIT = re.compile(r"[\d)]\s*[-+*/×÷x^]\s*[\d(]|\d+(?:\.\d+)?\s*%\s*of\s*\$?\d|\bsqrt\b|square root of\s*\d|"
                      r"\d\s*(?:times|plus|minus|divided by|multiplied by)\s*\$?\d", re.I)


def needs_calculator(text):
    """Offer the calculator only when the message asks for math: offered a tool, the model tends to call it for
    anything ("Hi!" got a calculation, "my dog is 3 years old" got calculate(type="toy", age=3)). Math means an
    arithmetic expression, or a number (digits or words) together with a math word."""
    has_number = bool(re.search(r"\d", text) or NUMBER_WORDS.search(text))
    return bool(ARITHMETIC.search(text)) or (has_number and bool(MATH_WORDS.search(text)))


def system_prompt(extra=""):
    """The model's system prompt: optional user instructions, the calculator rules and the function list."""
    head = "\n\n".join(p for p in (extra.strip(), CALC_INSTRUCTIONS) if p)
    return f"{head}\n\n{TOOLS_PROMPT.format(tools=json.dumps([TOOL]))}"


# ----------------------------------------------------------------------------- the calculator

_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
        ast.Mod: op.mod, ast.Pow: op.pow}
_UNARY = {ast.USub: op.neg, ast.UAdd: op.pos}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max, "floor": math.floor,
          "ceil": math.ceil}
_CONSTS = {"pi": math.pi, "e": math.e}
LIMIT = 1e18


def _normalize(expr):
    s = expr.strip().rstrip("=").strip()
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-").replace("^", "**").replace("$", "")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)                         # 1,234,567 -> 1234567
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\d(])", r"(\1/100)", s)    # 15% -> (15/100), but 7 % 3 stays modulo
    s = re.sub(r"\bof\b", "*", s)                                      # 15% of 80
    s = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", s)                     # 3 x 4 x 5
    return s


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(b) > 100 or abs(a) > 1e6):
            raise ValueError("power too large")
        v = _BIN[type(node.op)](a, b)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        v = _UNARY[type(node.op)](_eval(node.operand))
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        v = _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    elif isinstance(node, ast.Name) and node.id in _CONSTS:
        v = _CONSTS[node.id]
    else:
        raise ValueError("only numbers, + - * / // % **, parentheses and sqrt/abs/round/min/max/floor/ceil")
    if isinstance(v, complex) or abs(v) > LIMIT:
        raise ValueError("result out of range")
    return v


def calculate(expression):
    """{"expression", "result"} or {"expression", "error"}."""
    if not isinstance(expression, str) or len(expression) > 200:
        return {"expression": str(expression)[:200], "error": "the expression must be a short string"}
    try:
        v = _eval(ast.parse(_normalize(expression), mode="eval").body)
    except ZeroDivisionError:
        return {"expression": expression, "error": "division by zero"}
    except (SyntaxError, ValueError, TypeError, OverflowError) as e:
        return {"expression": expression, "error": str(e) or "invalid expression"}
    if isinstance(v, float):
        v = int(v) if v.is_integer() else float(f"{v:.10g}")
    return {"expression": expression, "result": v}


# ----------------------------------------------------------------------------- tool calls in model output

CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)


def parse_calls(text):
    """Expressions the model asked for, in order (any function name; we only have the one tool)."""
    out = []
    for body in CALL.findall(text):
        m = re.search(r"\{.*\}", body, re.S)
        if not m:
            continue
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            try:
                obj = ast.literal_eval(m.group(0))
            except (ValueError, SyntaxError):
                continue
        args = obj.get("arguments", obj) if isinstance(obj, dict) else {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {"expression": args}
        expr = args.get("expression") if isinstance(args, dict) else None
        if expr is None and isinstance(args, dict) and len(args) == 1:
            expr = next(iter(args.values()))
        if expr is not None:
            out.append(str(expr))
    return out


def strip_calls(text):
    return CALL.sub("", text).strip()


def tool_message(results):
    return {"role": "tool", "content": "\n".join(f"<tool_response>\n{json.dumps(r)}\n</tool_response>" for r in results)}


# ----------------------------------------------------------------------------- inline calculator

EXPR_EQ = re.compile(r"([0-9$.,()%+\-*/×x÷ ]+)=\s*$")
INLINE_OP = re.compile(r"[\d)]\s*[-+*/]\s*\(?\s*\d")


def _fmt(v):
    return str(v) if isinstance(v, int) else f"{v:.4f}".rstrip("0").rstrip(".")


def inline_calculator(tok, on_calc=None):
    """A generate(insert=...) callback: when the reply so far ends with an arithmetic expression and "=" (as in
    "16 - 3 - 4 ="), insert the exact result, so LibraMind keeps its own step-by-step reasoning but can't get the
    arithmetic wrong. Its word problems fail about equally from slips and from wrong reasoning; this fixes the slips.
    on_calc(result) is called for every inserted calculation."""
    def insert(row, ids):
        if "=" not in tok.decode(ids[-1:]):
            return None
        text = tok.decode(ids)
        m = EXPR_EQ.search(text)
        # judge it after normalizing, so "3 * $4.75 =" and "100% - 20% =" count as arithmetic
        if not m or not INLINE_OP.search(_normalize(m.group(1))):
            return None
        r = calculate(m.group(1).strip())
        if "result" not in r:
            return None
        if on_calc:
            on_calc(r)
        return tok.encode(("" if text.endswith(" ") else " ") + _fmt(r["result"]), add_special_tokens=False).ids
    return insert


# ----------------------------------------------------------------------------- the loop

def solve(model, enc, histories, system_extra="", temperature=0.0, top_p=1.0, rep_penalty=1.0, max_rounds=6,
          max_new=400, force_first=False, grammar=None, batch_size=64, on_event=None):
    """Answer many conversations with the calculator available. histories: lists of user/assistant messages.
    force_first: the first reply must be a calculator call (grammar-forced; needs `grammar`, a GrammarFactory).
    Returns [(final answer, [calculator results], full message list)]. on_event(i, kind, data) streams progress:
    ("calc", result) and ("final", text)."""
    import torch
    from inference import generate
    convs = [[{"role": "system", "content": system_prompt(system_extra)}] + list(h) for h in histories]
    traces, finals, pending = [[] for _ in convs], [None] * len(convs), list(range(len(convs)))

    def gen(rows, forced):
        prompts = [enc.encode_prompt(convs[i]) for i in rows]
        matchers = [grammar.tool_calls([TOOL]) for _ in rows] if forced and grammar else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = generate(model, prompts, max_new, temperature, top_p, rep_penalty=rep_penalty, stop_ids=[enc.im_end],
                            matchers=matchers, batch_size=batch_size, vocab=enc.tok.get_vocab_size())
        return {i: enc.tok.decode(o) for i, o in zip(rows, outs)}

    for rnd in range(max_rounds + 1):
        if not pending:
            break
        texts = gen(pending, force_first and rnd == 0)
        # a call the calculator can't evaluate (usually words instead of numbers): redo that step with the
        # expression grammar-constrained to numbers and operators, instead of letting the model guess
        broken = [i for i in pending if any("error" in calculate(e) for e in parse_calls(texts[i]))]
        if broken and grammar:
            texts.update(gen(broken, True))
        still = []
        for i in pending:
            text = texts[i]
            exprs = parse_calls(text)
            if exprs and rnd < max_rounds:
                results = [calculate(e) for e in exprs]
                traces[i] += results
                convs[i] += [{"role": "assistant", "content": text}, tool_message(results)]
                if on_event:
                    for r in results:
                        on_event(i, "calc", r)
                still.append(i)
            else:
                finals[i] = strip_calls(text)
                convs[i].append({"role": "assistant", "content": text})
        pending = still
    # a call with no usable expression (e.g. calculate(type="toy", age=3)) leaves nothing to show: answer those
    # conversations again without the calculator
    empty = [i for i in range(len(convs)) if not finals[i]]
    if empty:
        plain = [([{"role": "system", "content": system_extra}] if system_extra.strip() else []) + list(histories[i])
                 for i in empty]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = generate(model, [enc.encode_prompt(c) for c in plain], max_new, temperature, top_p,
                            rep_penalty=rep_penalty, stop_ids=[enc.im_end], batch_size=batch_size,
                            vocab=enc.tok.get_vocab_size())
        for i, o in zip(empty, outs):
            finals[i] = enc.tok.decode(o).strip()
            convs[i].append({"role": "assistant", "content": finals[i]})
    if on_event:
        for i in range(len(convs)):
            on_event(i, "final", finals[i])
    return [(finals[i], traces[i], convs[i]) for i in range(len(convs))]
