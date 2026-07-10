"""安全的数学表达式求值工具（用 ast 白名单，避免 eval 风险）。"""
import ast
import operator

from tools.base import Tool


class CalculatorTool(Tool):
    name = "calculator"
    description = "计算一个数学表达式，支持 + - * / ** 和括号。当用户需要算数时调用。"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "数学表达式，如 (2+3)*4"}
        },
        "required": ["expression"],
    }

    _OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def run(self, expression: str) -> str:
        try:
            value = self._eval(ast.parse(expression, mode="eval").body)
            return f"{expression} = {value}"
        except Exception as e:  # noqa: BLE001
            return f"无法计算 '{expression}': {e}"

    def _eval(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return self._OPS[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return self._OPS[type(node.op)](self._eval(node.operand))
        raise ValueError("不支持的表达式")
