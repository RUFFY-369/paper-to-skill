import ast

def validate_script_safety(code: str) -> bool:
    """
    Scans the untrusted candidate code via AST parsing and blocks disallowed nodes
    or dangerous system calls to enforce zero-trust execution.
    """
    # Block extremely dangerous keywords before even parsing
    dangerous_keywords = [
        "os.system", "subprocess.", "shutil.", "rmtree", "unlink",
        "__import__"
    ]
    for kw in dangerous_keywords:
        if kw in code:
            raise PermissionError(f"Security violation: dangerous keyword '{kw}' found in code.")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SyntaxError(f"Syntax error in synthesized code: {e}")

    # List of allowed standard and scientific modules
    allowed_modules = {
        "torch", "triton", "math", "typing", "sys", "inspect",
        "generated_sage2_skill", "sage2_harness", "fa3_harness", "fa3_runner"
    }

    # Inspect the AST tree for security violations
    for node in ast.walk(tree):
        # 1. Imports check
        if isinstance(node, ast.Import):
            for name in node.names:
                root_mod = name.name.split('.')[0]
                if root_mod not in allowed_modules:
                    raise PermissionError(f"Security violation: import of module '{root_mod}' is not allowed.")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_mod = node.module.split('.')[0]
                if root_mod not in allowed_modules:
                    raise PermissionError(f"Security violation: from-import of module '{root_mod}' is not allowed.")

        # 2. Prevent calls to builtins like open, eval, exec, __import__
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in {"eval", "exec", "open", "compile", "globals", "locals"}:
                    raise PermissionError(f"Security violation: restricted builtin '{node.func.id}' called.")
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id in {"os", "sys", "subprocess", "shutil"} and node.func.attr not in {"exit", "version"}:
                        raise PermissionError(f"Security violation: access to '{node.func.value.id}.{node.func.attr}' blocked.")

    return True
