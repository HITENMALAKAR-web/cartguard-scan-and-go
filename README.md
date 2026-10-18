# CartGuard — Vision-Assisted Verification for Mobile Scan-and-Go

> An independent vision layer that watches what goes into a customer's basket and checks whether the mobile cart (the bill) matches.

**Team:** TODO · **Hackathon:** TODO · **Demo video:** TODO

## The problem
Mobile Scan-and-Go lets customers skip the checkout queue, but it relies on them scanning every item. Missed scans (honest mistakes or not) become losses for the store, and forcing everyone through a traditional checkout removes the convenience.

## Our idea
CartGuard compares **what the camera sees in the basket** with **what the customer was billed for**, and flags the difference for review.

1. **Detect products and people** — a custom-trained YOLOv8n model recognises 200 retail product classes; a stock YOLOv8n model detects people. Both run with tracking.
2. **Build a vision basket** — a *basket zone* is drawn on the camera view. A product counts only while its centre is inside the zone. Counts are median-smoothed over about 25 frames and each track's class is stabilised by voting, so tracker ID switches and class flicker do not inflate the bill.
3. **Infer pickup / put-back** — when a class count rises the system logs a `PICKUP`; when it falls it logs a `PUT_BACK`.
4. **Reconcile** — the vision basket is compared item by item and by total with the checkout bill. Under-billed items (seen but not billed) and over-billed items (billed but not seen) are reported with their rupee value.

```
Camera -> YOLOv8 product model ─┐
                                ├-> basket zone + smoothing -> vision basket -> compare with bill -> alert
Camera -> YOLOv8 person model ──┘        (pickup / put-back events)
```

## Results

**Product detector** (YOLOv8n, validation split, from the training log in the notebook)

| Metric | Value |
|---|---|
| Classes | 200 |
| Validation images / instances | 1,928 / 9,825 |
| Precision | 0.839 |
| Recall | 0.840 |
| mAP@50 | 0.899 |
| mAP@50-95 | 0.772 |
| Model | 3.4M parameters, ~5.4 ms/image inference |
| Training | 30 epochs, 640 px, batch 8, ~1.2 h on an RTX 3050 6 GB laptop GPU |

These are detector metrics on the dataset's validation images, **not** end-to-end cart accuracy.

**Reconciliation example** (from the notebook): the camera's basket had 2× `180_tissue`, the test bill had 1×. Output:

```
product: 180_tissue x1 added to cart but not present at checkout
camera predicted higher bill than checkout by 60 and {'180_tissue': 1} were detected by machine but not present at checkout
```

## Repository layout
```
prototype.ipynb      # training, live basket, reconciliation
models/best.pt       # trained product detector weights
requirements.txt
```

## Setup and run
```bash
pip install -r requirements.txt
```
1. Put the trained weights at `models/best.pt`.
2. Open `prototype.ipynb` and run the cells in order. (Training needs the dataset at `grocery_datset/data.yaml`; skip the training cell to just use the provided weights.)
3. In the live cell, a window opens on the first camera frame: **drag a box around the basket and press ENTER.** Place items in the basket; the console prints `PICKUP` / `PUT_BACK` events and the window shows the running bill.
4. Press **c** to finish the basket (stored as `machine_output`), or **q** to quit.
5. Run the last cells to compare `machine_output` with the checkout dictionary.

To use a recorded clip instead of the webcam, replace `cv2.VideoCapture(0)` with `cv2.VideoCapture("clip.mp4")`.

## Known limitations
- **Noisy event log:** the pickup/put-back log flickers when detections briefly drop in and out (see the saved output). The smoothed final basket is much more stable than the raw event stream.
- **One fixed basket zone and a fixed camera.** Several shoppers or moving carts would need one zone per basket or a zone attached to each person.
- **Person detection is displayed but not yet used for billing or basket ownership.**
- **The mobile cart is simulated** with a dictionary, and prices come from a demo catalogue table.
- Items stacked on top of each other can hide one another and be undercounted.
- No end-to-end accuracy study yet; only a small number of manual tests.

## Roadmap
- Tie products to the hand holding them with pose keypoints (wrists) to improve pickup/put-back and person assignment.
- Debounce the event log and add per-person baskets.
- Connect to a real mobile-app cart API and run an alert/verification flow.
- Evaluate end to end on recorded shopping sessions.

## Dataset and licence
- Dataset: TODO (name, source link, licence).
- This project uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), which is licensed under AGPL-3.0; choose this repository's licence accordingly. Licence: TODO.
