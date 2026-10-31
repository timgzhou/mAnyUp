"""Preview sheet of method_parts stacks at one scale: python -m exp.viz.paper.method_preview <setup> <out.png> <tiles...> (run from figures/paper/method)."""
import sys
from PIL import Image, ImageDraw
setup, out, tiles = sys.argv[1], sys.argv[2], sys.argv[3:]
names=["stack_months","stack_seasons","stack_lr_features","stack_selection","stack_hr_tokens_active","stack_month_features_active","stack_mixed_features_active","single/label","single/pred_final"]
s, gap = 0.25, 16      # one scale for every stack (frames the same size), px between columns
ims = [[Image.open(f"{setup}/{d.replace('@', '_')}/{n}.png").convert("RGBA") for n in names]
       for d in tiles]
ims = [[im.resize((max(1, int(im.width * s)), max(1, int(im.height * s)))) if "stack" in n
        else im.resize((110, 110)) for im, n in zip(row, names)] for row in ims]
cw = [max(row[k].width for row in ims) + gap for k in range(len(names))]
rh = [max(im.height for im in row) + gap for row in ims]
sheet = Image.new("RGBA", (sum(cw), sum(rh)), (255, 255, 255, 255))
y = 0
for d, row, h in zip(tiles, ims, rh):
    x = 0
    for im, w in zip(row, cw):
        sheet.alpha_composite(im, (x, y + h - gap - im.height))
        x += w
    ImageDraw.Draw(sheet).text((2, y + 2), d, fill=(200, 0, 0, 255))
    y += h
sheet.convert("RGB").save(out)
