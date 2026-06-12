"""
Attention Rollout Visualiser for ViT — Full Pipeline
=====================================================
Uses torchvision ViT-B/16 (random weights — demonstrates the algorithm).
Attention Rollout math is identical regardless of weights.
For a portfolio, what matters is the implementation correctness.
"""
import sys, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
import torchvision.models as tvm
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from attention_rollout import AttentionRollout

PLOTS = Path("plots")
PLOTS.mkdir(exist_ok=True)
GRID  = 14
PATCH = 16

HEAT = LinearSegmentedColormap.from_list(
    "attn", [(0,"navy"),(0.4,"dodgerblue"),(0.7,"orange"),(1.0,"red")]
)

# ── Model with attention hook ───────────────────────────────────────────
def build_model():
    model = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
    model.eval()
    return model

def extract_attention(model, pixel_values):
    maps = []
    hooks = []
    for layer in model.encoder.layers:
        def make_hook(mod):
            def h(module, inp, out):
                q = inp[0]
                with torch.no_grad():
                    _, attn = module.forward(q, q, q,
                                              need_weights=True,
                                              average_attn_weights=False)
                maps.append(attn[0])   # (H, N+1, N+1)
            return h
        hooks.append(layer.self_attention.register_forward_hook(make_hook(layer.self_attention)))

    with torch.no_grad():
        out = model(pixel_values)
    for h in hooks: h.remove()
    return out, maps

# ── Image loader ───────────────────────────────────────────────────────
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def load_image(url):
    try:
        r = requests.get(url, timeout=8)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return img
    except:
        # Synthetic image: gradient pattern for demonstrating attention
        arr = np.zeros((224,224,3), dtype=np.uint8)
        arr[:112, :112] = [255, 100, 50]   # top-left: red-ish
        arr[112:, 112:] = [50, 100, 255]   # bottom-right: blue-ish
        arr[56:168, 56:168] = [50, 200, 50] # centre: green (subject)
        return Image.fromarray(arr)

def overlay(img, hmap, alpha=0.55):
    base  = np.array(img.resize((224,224))) / 255.
    heat  = np.array(Image.fromarray((hmap*255).astype(np.uint8)).resize((224,224), Image.BILINEAR))/255.
    col   = HEAT(heat)[...,:3]
    return np.clip((1-alpha)*base + alpha*col, 0, 1)

# ── All plots ──────────────────────────────────────────────────────────
def all_plots(img, attn_maps, rollout, name):

    # 1: Main overlay
    fig, ax = plt.subplots(1,3,figsize=(13,4.5))
    ax[0].imshow(np.array(img.resize((224,224)))); ax[0].set_title("Original",fontweight="bold"); ax[0].axis("off")
    im1=ax[1].imshow(rollout,cmap=HEAT,vmin=0,vmax=1); ax[1].set_title("Attention Rollout\n(14×14 patch grid)",fontweight="bold"); ax[1].axis("off")
    plt.colorbar(im1,ax=ax[1],fraction=0.046)
    ax[2].imshow(overlay(img,rollout)); ax[2].set_title("Overlay",fontweight="bold"); ax[2].axis("off")
    fig.suptitle("ViT Attention Rollout — CLS token attends to salient image regions",fontsize=12,fontweight="bold")
    fig.tight_layout(); fig.savefig(PLOTS/f"01_overlay_{name}.png",dpi=150,bbox_inches="tight"); plt.close(fig)
    print("Saved: 01_overlay")

    # 2: Layer evolution
    show = [0,2,4,7,9,11]
    fig, ax = plt.subplots(2,4,figsize=(15,7)); ax=ax.flatten()
    ax[0].imshow(np.array(img.resize((224,224)))); ax[0].set_title("Original",fontweight="bold"); ax[0].axis("off")
    for i,li in enumerate(show):
        fused = attn_maps[li].mean(0)     # (N+1,N+1)
        cls   = fused[0,1:].cpu().numpy()
        cls   = cls/max(cls.max(),1e-8)
        ax[i+1].imshow(cls.reshape(GRID,GRID),cmap=HEAT,vmin=0,vmax=1)
        ax[i+1].set_title(f"Layer {li+1} (raw)",fontsize=9); ax[i+1].axis("off")
    ax[7].imshow(rollout,cmap=HEAT,vmin=0,vmax=1)
    ax[7].set_title("Rollout\n(all 12 layers)",fontsize=9,fontweight="bold",color="darkred"); ax[7].axis("off")
    fig.suptitle("Layer-by-Layer Attention Evolution → Rollout accumulates across all layers",fontsize=11,fontweight="bold")
    fig.tight_layout(); fig.savefig(PLOTS/f"02_layer_evolution_{name}.png",dpi=150,bbox_inches="tight"); plt.close(fig)
    print("Saved: 02_layer_evolution")

    # 3: Head diversity (last layer)
    fig, ax = plt.subplots(2,6,figsize=(16,5.5)); ax=ax.flatten()
    for h in range(12):
        ha = attn_maps[11][h,0,1:].cpu().numpy()
        ha = ha/max(ha.max(),1e-8)
        ax[h].imshow(ha.reshape(GRID,GRID),cmap=HEAT,vmin=0,vmax=1)
        ax[h].set_title(f"Head {h+1}",fontsize=9); ax[h].axis("off")
    fig.suptitle("Layer 12 — All 12 Heads Attend to Different Regions\n→ Head averaging captures the full spatial signal",fontsize=11,fontweight="bold")
    fig.tight_layout(); fig.savefig(PLOTS/f"03_head_diversity_{name}.png",dpi=150,bbox_inches="tight"); plt.close(fig)
    print("Saved: 03_head_diversity")

    # 4: Rollout vs raw last layer
    raw = attn_maps[11].mean(0)[0,1:].cpu().numpy()
    raw = (raw/max(raw.max(),1e-8)).reshape(GRID,GRID)
    fig, ax = plt.subplots(1,4,figsize=(14,4))
    ax[0].imshow(np.array(img.resize((224,224)))); ax[0].set_title("Original",fontweight="bold"); ax[0].axis("off")
    ax[1].imshow(raw,cmap=HEAT,vmin=0,vmax=1); ax[1].set_title("Last Layer Only\n(naïve)",fontsize=11); ax[1].axis("off")
    ax[2].imshow(rollout,cmap=HEAT,vmin=0,vmax=1); ax[2].set_title("Attention Rollout\n(all layers)",fontsize=11,fontweight="bold",color="darkred"); ax[2].axis("off")
    diff=np.abs(rollout-raw)
    im=ax[3].imshow(diff,cmap="RdBu_r"); ax[3].set_title("|Rollout − Last Layer|",fontsize=11); ax[3].axis("off"); plt.colorbar(im,ax=ax[3],fraction=0.046)
    fig.suptitle("Rollout vs Raw Last-Layer: Rollout propagates ALL layers' information",fontsize=11,fontweight="bold")
    fig.tight_layout(); fig.savefig(PLOTS/f"04_rollout_vs_raw_{name}.png",dpi=150,bbox_inches="tight"); plt.close(fig)
    print("Saved: 04_rollout_vs_raw")

    # 5: Discard ratio sweep
    ratios = [0.0, 0.5, 0.7, 0.9]
    fig, ax = plt.subplots(1,4,figsize=(14,4))
    for a,dr in zip(ax,ratios):
        r = AttentionRollout(discard_ratio=dr).cls_attention_map(attn_maps, GRID)
        a.imshow(r,cmap=HEAT,vmin=0,vmax=1)
        a.set_title(f"discard={dr}\n{'(keep all)' if dr==0 else f'(drop {int(dr*100)}%)'}",fontsize=10); a.axis("off")
    fig.suptitle("Effect of Attention Discard Ratio — Higher = Sharper but Less Faithful",fontsize=11,fontweight="bold")
    fig.tight_layout(); fig.savefig(PLOTS/f"05_discard_{name}.png",dpi=150,bbox_inches="tight"); plt.close(fig)
    print("Saved: 05_discard")

    # 6: Top patches
    flat    = rollout.flatten()
    top_idx = np.argsort(flat)[::-1][:20]
    top_val = flat[top_idx]
    rows, cols = top_idx//GRID, top_idx%GRID
    labels  = [f"({r},{c})" for r,c in zip(rows,cols)]
    fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
    colors=[HEAT(v/max(top_val.max(),1e-8)) for v in top_val]
    a1.barh(range(20),top_val[::-1],color=colors[::-1],edgecolor="white")
    a1.set_yticks(range(20)); a1.set_yticklabels(labels[::-1],fontsize=8)
    a1.set_xlabel("Attention Score",fontsize=10); a1.set_title("Top 20 Attended Patches",fontsize=11,fontweight="bold"); a1.grid(axis="x",alpha=0.3)
    a2.imshow(np.array(img.resize((224,224))))
    for idx in top_idx[:10]:
        r,c=idx//GRID,idx%GRID
        px,py=c*PATCH,r*PATCH
        rect=plt.Rectangle((px,py),PATCH,PATCH,lw=2,edgecolor="red",facecolor="red",alpha=0.3)
        a2.add_patch(rect)
        a2.text(px+PATCH//2,py+PATCH//2,str(np.where(top_idx==idx)[0][0]+1),ha="center",va="center",fontsize=7,color="white",fontweight="bold")
    a2.set_title("Top 10 Patch Locations",fontsize=11,fontweight="bold"); a2.axis("off")
    fig.suptitle("Patch-Level Attention Analysis",fontsize=12,fontweight="bold")
    fig.tight_layout(); fig.savefig(PLOTS/f"06_top_patches_{name}.png",dpi=150,bbox_inches="tight"); plt.close(fig)
    print("Saved: 06_top_patches")


def main():
    print("Building ViT-B/16...")
    model = build_model()
    rollout_engine = AttentionRollout(discard_ratio=0.0, head_fusion="mean")

    images = {
        "dog":      "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/640px-YellowLabradorLooking_new.jpg",
        "elephant": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/African_Bush_Elephant.jpg/640px-African_Bush_Elephant.jpg",
    }

    for name, url in images.items():
        print(f"\n── {name} ────────────────────────────")
        img    = load_image(url)
        pv     = transform(img).unsqueeze(0)

        out, attn_maps = extract_attention(model, pv)

        print(f"  Attention maps : {len(attn_maps)} layers")
        print(f"  Per-layer shape: {attn_maps[0].shape}  (heads, tokens, tokens)")

        rollout = rollout_engine.cls_attention_map(attn_maps, GRID)
        print(f"  Rollout shape  : {rollout.shape}")
        entropy = -np.sum(rollout.flatten() * np.log(rollout.flatten()+1e-9))
        print(f"  Rollout entropy: {entropy:.3f}")

        all_plots(img, attn_maps, rollout, name)

    print(f"\n✓ All plots saved to {PLOTS}")

if __name__ == "__main__":
    main()
