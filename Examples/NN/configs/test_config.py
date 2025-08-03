import yaml

with open("test.yaml") as f:
    data = yaml.safe_load(f)

print(type(data["a"]))  # <class 'float'>
print(type(data["b"]))  # <class 'str'>
print(type(data["c"]))  # <class 'str'>
