from pathlib import Path

a = Path("test/test.py")
p = Path("out")

# 输出 out/test.vocals.wav

print(p / a.with_suffix(".vocals.wav").name)