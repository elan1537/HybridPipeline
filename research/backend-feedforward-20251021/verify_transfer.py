import os
import hashlib

def md5(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

renderer_dir = "/home/wrl-ubuntu/workspace/HybridPipeline/research/3DGStream/renderer_output"
transport_dir = "transport_output"

renderer_files = sorted([f for f in os.listdir(renderer_dir) if f.endswith('.jpg')])
transport_files = sorted([f for f in os.listdir(transport_dir) if f.endswith('.jpg')])

print(f"Renderer: {len(renderer_files)} files")
print(f"Transport: {len(transport_files)} files")

mismatches = []
for fname in renderer_files:
    if fname not in transport_files:
        print(f"Missing in transport: {fname}")
        continue

    r_hash = md5(os.path.join(renderer_dir, fname))
    t_hash = md5(os.path.join(transport_dir, fname))

    if r_hash != t_hash:
        mismatches.append(fname)
        print(f"Mismatch: {fname}")
        print(f"  Renderer:  {r_hash}")
        print(f"  Transport: {t_hash}")

if not mismatches:
    print("\n✅ All files match! Transfer successful.")
else:
    print(f"\n❌ {len(mismatches)} files mismatched")
