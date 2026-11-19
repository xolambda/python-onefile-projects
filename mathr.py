import math
import re
from fractions import Fraction
from functools import reduce
from math import gcd, lcm


FUNCTIONS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sqrt": math.sqrt,
    "log": math.log,
    "ln": math.log,
    "exp": math.exp,
    "abs": abs
}

CONSTANTS = {
    "pi": math.pi,
    "e": math.e
}

TOKEN_PATTERN = re.compile(
    r"\s*(?:(\d+(?:\.\d*)?|\.\d+)|([a-zA-Z_]+)|(\*\*|[+\-*/^(),=]))"
)


class MathError(Exception):
    pass


def number(value):
    return ("num", Fraction(value))


def tokenize(expression):
    tokens = []
    position = 0

    while position < len(expression):
        if not expression[position:].strip():
            break

        match = TOKEN_PATTERN.match(expression, position)

        if not match:
            raise MathError(f"Unexpected character: {expression[position]}")

        numeric, identifier, operator = match.groups()

        if numeric:
            tokens.append(("number", numeric))
        elif identifier:
            tokens.append(("identifier", identifier.lower()))
        else:
            tokens.append(("operator", operator))

        position = match.end()

    return tokens


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.position = 0

    def peek(self):
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]

    def consume(self):
        token = self.peek()

        if token is None:
            raise MathError("Unexpected end of expression")

        self.position += 1
        return token

    def expression(self, minimum_binding=0):
        kind, value = self.consume()

        if kind == "number":
            left = number(value)

        elif kind == "identifier":
            if value in FUNCTIONS and self.peek() == ("operator", "("):
                self.consume()
                argument = self.expression()

                if self.consume() != ("operator", ")"):
                    raise MathError("Expected closing parenthesis")

                left = ("func", value, argument)
            else:
                left = ("var", value)

        elif kind == "operator" and value == "(":
            left = self.expression()

            if self.consume() != ("operator", ")"):
                raise MathError("Expected closing parenthesis")

        elif kind == "operator" and value in ("+", "-"):
            operand = self.expression(25)
            left = operand if value == "+" else (
                "mul", number(-1), operand
            )

        else:
            raise MathError(f"Unexpected token: {value}")

        while True:
            token = self.peek()

            if token is None:
                break

            if token[0] == "operator" and token[1] in (")", ",", "="):
                break

            op = token[1]

            if token[0] == "operator" and op in ("+", "-"):
                lbp, rbp = 10, 11
            elif token[0] == "operator" and op in ("*", "/"):
                lbp, rbp = 20, 21
            elif token[0] == "operator" and op in ("^", "**"):
                lbp, rbp = 30, 30
            elif token[0] in ("number", "identifier") or (
                token[0] == "operator" and op == "("
            ):
                op = "*"
                lbp, rbp = 20, 21
            else:
                break

            if lbp < minimum_binding:
                break

            if token[0] == "operator" and token[1] != "(":
                self.consume()

            right = self.expression(rbp)

            if op == "+":
                left = ("add", left, right)
            elif op == "-":
                left = ("add", left, ("mul", number(-1), right))
            elif op == "*":
                left = ("mul", left, right)
            elif op == "/":
                left = ("div", left, right)
            else:
                left = ("pow", left, right)

        return left

    def parse(self):
        result = self.expression()

        if self.peek() is not None:
            raise MathError(f"Unexpected token: {self.peek()[1]}")

        return simplify(result)


def simplify(node):
    kind = node[0]

    if kind in ("num", "var"):
        return node

    if kind == "func":
        argument = simplify(node[2])

        if argument[0] == "num":
            try:
                result = FUNCTIONS[node[1]](float(argument[1]))

                if math.isfinite(result):
                    return number(Fraction(str(result)))
            except (ValueError, OverflowError, ZeroDivisionError):
                pass

        return ("func", node[1], argument)

    left = simplify(node[1])
    right = simplify(node[2])

    if kind == "add":
        if left[0] == right[0] == "num":
            return number(left[1] + right[1])
        if left == number(0):
            return right
        if right == number(0):
            return left

    elif kind == "mul":
        if left[0] == right[0] == "num":
            return number(left[1] * right[1])
        if left == number(0) or right == number(0):
            return number(0)
        if left == number(1):
            return right
        if right == number(1):
            return left

    elif kind == "div":
        if right == number(0):
            raise MathError("Division by zero")
        if left[0] == right[0] == "num":
            return number(left[1] / right[1])
        if left == number(0):
            return number(0)
        if right == number(1):
            return left

    elif kind == "pow":
        if right == number(0):
            return number(1)
        if right == number(1):
            return left
        if left == number(1):
            return number(1)

        if left[0] == right[0] == "num":
            if right[1].denominator == 1:
                try:
                    return number(left[1] ** int(right[1]))
                except ZeroDivisionError:
                    raise MathError("Division by zero")

    return (kind, left, right)


def variables_in(node):
    if node[0] == "var":
        return {node[1]} if node[1] not in CONSTANTS else set()

    if node[0] in ("num",):
        return set()

    if node[0] == "func":
        return variables_in(node[2])

    result = variables_in(node[1])
    result.update(variables_in(node[2]))
    return result


def detect_variable(*nodes):
    variables = set()

    for node in nodes:
        variables.update(variables_in(node))

    if len(variables) > 1:
        raise MathError("Multiple variables are not supported yet")

    return next(iter(variables), "x")


def polynomial(node, variable="x"):
    kind = node[0]

    if kind == "num":
        return {0: node[1]}

    if kind == "var":
        if node[1] != variable:
            raise MathError("Unexpected variable")
        return {1: Fraction(1)}

    if kind == "add":
        a = polynomial(node[1], variable)
        b = polynomial(node[2], variable)
        result = dict(a)

        for power, coefficient in b.items():
            result[power] = result.get(power, Fraction(0)) + coefficient

        return {p: c for p, c in result.items() if c}

    if kind == "mul":
        a = polynomial(node[1], variable)
        b = polynomial(node[2], variable)
        result = {}

        for p1, c1 in a.items():
            for p2, c2 in b.items():
                p = p1 + p2
                result[p] = result.get(p, Fraction(0)) + c1 * c2

        return {p: c for p, c in result.items() if c}

    if kind == "div":
        numerator = polynomial(node[1], variable)
        denominator = simplify(node[2])

        if denominator[0] != "num" or denominator[1] == 0:
            raise MathError("Polynomial division requires a nonzero constant")

        return {
            p: c / denominator[1]
            for p, c in numerator.items()
        }

    if kind == "pow":
        exponent = simplify(node[2])

        if exponent[0] != "num" or exponent[1].denominator != 1:
            raise MathError("Polynomial powers must be integers")

        power = int(exponent[1])

        if power < 0:
            raise MathError("Negative polynomial powers are unsupported")

        base = polynomial(node[1], variable)
        result = {0: Fraction(1)}

        for _ in range(power):
            multiplied = {}

            for p1, c1 in result.items():
                for p2, c2 in base.items():
                    p = p1 + p2
                    multiplied[p] = multiplied.get(p, Fraction(0)) + c1 * c2

            result = multiplied

        return {p: c for p, c in result.items() if c}

    raise MathError("Expression is not a polynomial")


def format_number(value):
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"

    return str(value)


def format_polynomial(poly, variable="x"):
    if not poly:
        return "0"

    pieces = []

    for power in sorted(poly, reverse=True):
        coefficient = poly[power]

        if not coefficient:
            continue

        negative = coefficient < 0
        magnitude = abs(coefficient)

        if power == 0:
            body = format_number(magnitude)
        else:
            variable_part = variable if power == 1 else f"{variable}^{power}"
            body = variable_part if magnitude == 1 else (
                f"{format_number(magnitude)}{variable_part}"
            )

        if not pieces:
            pieces.append(("-" if negative else "") + body)
        else:
            pieces.append((" - " if negative else " + ") + body)

    return "".join(pieces) if pieces else "0"


def format_expression(node, variable="x"):
    try:
        return format_polynomial(polynomial(node, variable), variable)
    except MathError:
        pass

    kind = node[0]

    if kind == "num":
        return format_number(node[1])

    if kind == "var":
        return node[1]

    if kind == "func":
        return f"{node[1]}({format_expression(node[2], variable)})"

    if kind == "add":
        return f"({format_expression(node[1], variable)} + {format_expression(node[2], variable)})"

    if kind == "mul":
        return f"({format_expression(node[1], variable)} * {format_expression(node[2], variable)})"

    if kind == "div":
        return f"({format_expression(node[1], variable)} / {format_expression(node[2], variable)})"

    if kind == "pow":
        return f"({format_expression(node[1], variable)}^{format_expression(node[2], variable)})"

    return str(node)


def polynomial_content(poly):
    coefficients = [c for c in poly.values() if c]

    if not coefficients:
        return Fraction(1)

    numerator_gcd = reduce(
        gcd,
        (abs(c.numerator) for c in coefficients)
    )

    denominator_lcm = reduce(
        lcm,
        (c.denominator for c in coefficients)
    )

    return Fraction(numerator_gcd, denominator_lcm)


def integer_divisors(n):
    n = abs(int(n))

    if n == 0:
        return [0]

    result = set()

    for i in range(1, math.isqrt(n) + 1):
        if n % i == 0:
            result.add(i)
            result.add(n // i)

    return sorted(result)


def synthetic_division(poly, root):
    degree = max(poly)

    coefficients = [
        poly.get(p, Fraction(0))
        for p in range(degree, -1, -1)
    ]

    output = [coefficients[0]]

    for coefficient in coefficients[1:]:
        output.append(coefficient + root * output[-1])

    remainder = output.pop()

    quotient = {
        degree - 1 - i: coefficient
        for i, coefficient in enumerate(output)
        if coefficient
    }

    return quotient, remainder


def rational_roots(poly):
    degree = max(poly)
    constant = poly.get(0, Fraction(0))
    leading = poly[degree]

    if constant == 0:
        return [Fraction(0)]

    scale = reduce(
        lcm,
        (c.denominator for c in poly.values()),
        1
    )

    integer_poly = {
        p: int(c * scale)
        for p, c in poly.items()
    }

    numerators = integer_divisors(integer_poly.get(0, 0))
    denominators = integer_divisors(integer_poly[degree])

    candidates = set()

    for numerator in numerators:
        for denominator in denominators:
            if denominator:
                candidates.add(Fraction(numerator, denominator))
                candidates.add(Fraction(-numerator, denominator))

    return sorted(candidates)


def factor_polynomial(poly, variable="x"):
    poly = {p: Fraction(c) for p, c in poly.items() if c}

    if not poly:
        return "0"

    content = polynomial_content(poly)
    primitive = {p: c / content for p, c in poly.items()}

    minimum_power = min(primitive)

    primitive = {
        p - minimum_power: c
        for p, c in primitive.items()
    }

    roots = []

    while primitive and max(primitive) > 0:
        found = None

        for root in rational_roots(primitive):
            quotient, remainder = synthetic_division(primitive, root)

            if remainder == 0:
                found = (root, quotient)
                break

        if found is None:
            break

        root, primitive = found
        roots.append(root)

    denominator_product = math.prod(
        (root.denominator for root in roots),
        start=1
    )

    factors = []

    if minimum_power:
        factors.append(
            variable if minimum_power == 1
            else f"{variable}^{minimum_power}"
        )

    for root in roots:
        denominator = root.denominator
        numerator = -root.numerator

        if denominator == 1:
            if numerator == 0:
                factor = variable
            elif numerator > 0:
                factor = f"({variable} + {numerator})"
            else:
                factor = f"({variable} - {abs(numerator)})"
        else:
            first = (
                variable if denominator == 1
                else f"{denominator}{variable}"
            )

            if numerator > 0:
                factor = f"({first} + {numerator})"
            elif numerator < 0:
                factor = f"({first} - {abs(numerator)})"
            else:
                factor = f"({first})"

        factors.append(factor)

    if max(primitive) == 0:
        scalar = content * primitive.get(0, Fraction(0))
        scalar /= denominator_product

        if scalar == 0:
            return "0"

        if scalar != 1:
            if scalar == -1:
                prefix = "-"
            else:
                prefix = format_number(scalar)

            factors.insert(0, prefix)

    else:
        scalar = content / denominator_product

        remainder = {
            p: c * scalar
            for p, c in primitive.items()
        }

        remaining = format_polynomial(remainder, variable)

        factors.insert(0, f"({remaining})")

    return "".join(factors) if factors else "1"


def differentiate(node, variable="x"):
    kind = node[0]

    if kind == "num":
        return number(0)

    if kind == "var":
        return number(1 if node[1] == variable else 0)

    if kind == "add":
        return simplify((
            "add",
            differentiate(node[1], variable),
            differentiate(node[2], variable)
        ))

    if kind == "mul":
        left, right = node[1], node[2]

        return simplify((
            "add",
            ("mul", differentiate(left, variable), right),
            ("mul", left, differentiate(right, variable))
        ))

    if kind == "div":
        left, right = node[1], node[2]

        numerator = (
            "add",
            ("mul", differentiate(left, variable), right),
            ("mul", number(-1), (
                "mul", left, differentiate(right, variable)
            ))
        )

        return simplify((
            "div",
            numerator,
            ("pow", right, number(2))
        ))

    if kind == "pow":
        base, exponent = node[1], node[2]

        if exponent[0] == "num":
            return simplify((
                "mul",
                ("mul", exponent, (
                    "pow", base, number(exponent[1] - 1)
                )),
                differentiate(base, variable)
            ))

        raise MathError("Variable exponents are not supported yet")

    if kind == "func":
        name, argument = node[1], node[2]
        derivative = differentiate(argument, variable)

        if name == "sin":
            outer = ("func", "cos", argument)
        elif name == "cos":
            outer = ("mul", number(-1), ("func", "sin", argument))
        elif name == "tan":
            outer = ("div", number(1), (
                "pow", ("func", "cos", argument), number(2)
            ))
        elif name == "exp":
            outer = ("func", "exp", argument)
        elif name in ("log", "ln"):
            outer = ("div", number(1), argument)
        elif name == "sqrt":
            outer = ("div", number(1), (
                "mul", number(2), ("func", "sqrt", argument)
            ))
        else:
            raise MathError(f"Cannot differentiate {name}")

        return simplify(("mul", outer, derivative))

    raise MathError("Unsupported differentiation operation")


def integrate(node, variable="x"):
    poly = polynomial(node, variable)

    result = {
        power + 1: coefficient / (power + 1)
        for power, coefficient in poly.items()
    }

    return result


def evaluate(node, variables=None):
    variables = variables or {}
    kind = node[0]

    if kind == "num":
        return node[1]

    if kind == "var":
        if node[1] in variables:
            return variables[node[1]]

        if node[1] in CONSTANTS:
            return CONSTANTS[node[1]]

        raise MathError(f"Unknown variable: {node[1]}")

    if kind == "add":
        return evaluate(node[1], variables) + evaluate(node[2], variables)

    if kind == "mul":
        return evaluate(node[1], variables) * evaluate(node[2], variables)

    if kind == "div":
        return evaluate(node[1], variables) / evaluate(node[2], variables)

    if kind == "pow":
        return evaluate(node[1], variables) ** evaluate(node[2], variables)

    if kind == "func":
        return FUNCTIONS[node[1]](float(evaluate(node[2], variables)))

    raise MathError("Cannot evaluate expression")


def solve_equation(left, right, variable="x"):
    a = polynomial(left, variable)
    b = polynomial(right, variable)

    equation = dict(a)

    for power, coefficient in b.items():
        equation[power] = equation.get(power, Fraction(0)) - coefficient

    equation = {p: c for p, c in equation.items() if c}

    if not equation:
        return "Every value is a solution"

    degree = max(equation)

    if degree == 0:
        return "No solution"

    if degree == 1:
        return [-equation.get(0, 0) / equation[1]]

    if degree == 2:
        a = equation.get(2, Fraction(0))
        b = equation.get(1, Fraction(0))
        c = equation.get(0, Fraction(0))

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            real = float(-b / (2 * a))
            imaginary = math.sqrt(float(-discriminant)) / float(2 * a)

            return [
                complex(real, imaginary),
                complex(real, -imaginary)
            ]

        root = math.sqrt(float(discriminant))

        return [
            (-float(b) + root) / (2 * float(a)),
            (-float(b) - root) / (2 * float(a))
        ]

    raise MathError("Equation solver currently supports degrees up to 2")


def normalize_question(question):
    question = question.lower().strip()

    replacements = {
        "×": "*",
        "÷": "/",
        "−": "-",
        "²": "^2",
        "³": "^3",
        "⁴": "^4",
        "π": "pi",
        "squared": "^2",
        "cubed": "^3",
        "multiplied by": "*",
        "divided by": "/",
        "plus": "+",
        "minus": "-",
        "times": "*"
    }

    for old, new in replacements.items():
        question = question.replace(old, new)

    for word in ("what is", "calculate", "compute", "find", "the", "of"):
        question = re.sub(r"\b" + word + r"\b", "", question)

    return re.sub(r"\s+", " ", question).strip()


def parse_expression(text):
    return Parser(tokenize(text)).parse()


def solve_question(original):
    question = normalize_question(original)

    explicit_variable = re.search(
        r"\bwith respect to\s+([a-zA-Z])",
        question
    )

    question = re.sub(
        r"\bwith respect to\s+[a-zA-Z]\b",
        "",
        question
    ).strip()

    operations = (
        "factorise", "factorize", "factor",
        "differentiate", "derivative",
        "integrate", "simplify", "expand",
        "solve", "evaluate"
    )

    operation = None

    for candidate in operations:
        if question.startswith(candidate + " "):
            operation = candidate
            question = question[len(candidate):].strip()
            break

    if operation == "solve" and "=" not in question:
        question += "=0"

    if "=" in question:
        left_text, right_text = question.split("=", 1)
        left = parse_expression(left_text)
        right = parse_expression(right_text)

        variable = (
            explicit_variable.group(1).lower()
            if explicit_variable
            else detect_variable(left, right)
        )

        return solve_equation(left, right, variable)

    expression = parse_expression(question)

    variable = (
        explicit_variable.group(1).lower()
        if explicit_variable
        else detect_variable(expression)
    )

    if operation in ("factorise", "factorize", "factor"):
        return factor_polynomial(
            polynomial(expression, variable),
            variable
        )

    if operation == "differentiate" or operation == "derivative":
        return format_expression(
            differentiate(expression, variable),
            variable
        )

    if operation == "integrate":
        return format_polynomial(
            integrate(expression, variable),
            variable
        ) + " + C"

    if operation in ("simplify", "expand"):
        return format_polynomial(
            polynomial(expression, variable),
            variable
        )

    if operation == "evaluate":
        return format_number(evaluate(expression))

    try:
        return format_number(evaluate(expression))
    except MathError:
        return format_expression(expression, variable)


def main():
    print()
    print("Mathr")
    print("Don't use this for homework. seriously.")
    print("Type 'help' for commands or 'exit' to quit.")
    print()

    while True:
        try:
            question = input("You: ").strip()

            if not question:
                continue

            if question.lower() in ("exit", "quit"):
                print("Goodbye.")
                break

            if question.lower() == "help":
                print("""
Arithmetic:
  2 + 2
  sqrt(144)
  sin(pi / 2)

Algebra:
  simplify 2x + 3x + 4
  expand (x + 2)(x + 3)
  factorise x^2 + 5x + 6
  factorise 1 - x^2
  factorise 6y^2 + 13y + 5

Equations:
  solve x^2 - 5x + 6 = 0

Calculus:
  differentiate 3x^3 + 2x
  integrate 3x^2

Variables:
  x, y, z and other single-letter variables

Functions:
  sin, cos, tan, asin, acos, atan
  sqrt, log, ln, exp, abs

Use ^ for powers.
""")
                continue

            print("Solver:", solve_question(question))

        except (MathError, ValueError, ZeroDivisionError, OverflowError) as error:
            print("Error:", error)

        except KeyboardInterrupt:
            print("\nGoodbye.")
            break


if __name__ == "__main__":
    main()
