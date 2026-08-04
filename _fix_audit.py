with open('src/reproagent/audit.py', 'r') as f:
    content = f.read()

# Extract the try block using string slicing
start = '    try:\n        import torch\n'
end = '    print(json.dumps(data, indent=2, ensure_ascii=False))\n'

s = content.find(start)
e = content.find(end, s)
if s < 0 or e < 0:
    print(f'NOT FOUND: s={s}, e={e}')
    exit(1)

old_block = content[s:e + len(end)]

new_block = '''    for lib, check_gpu in (("torch", True), ("tensorflow", False), ("jax", False)):
        try:
            mod = __import__(lib)
            info = {"version": getattr(mod, "__version__", None)}
            if check_gpu:
                info["cuda_compiled"] = getattr(getattr(mod, "version", None), "cuda", None)
                info["cuda_available"] = bool(mod.cuda.is_available())
                info["device_count"] = mod.cuda.device_count()
            data[lib] = info
        except Exception:
            pass
    print(json.dumps(data, indent=2, ensure_ascii=False))
'''

content = content.replace(old_block, new_block)
with open('src/reproagent/audit.py', 'w') as f:
    f.write(content)
print(f'OK, replaced {len(old_block)} chars with {len(new_block)} chars')
