# CartGuard — Vision-Assisted Verification for Mobile Scan-and-Go

> An independent vision layer that watches what each customer puts into their basket and checks whether their mobile cart (the bill) matches.

**Team:** CARTGUARD

## The problem
Mobile Scan-and-Go lets customers skip the checkout queue, but it relies on them scanning every item. Missed scans, honest or not, become losses for the store, and forcing everyone through a traditional checkout removes the convenience.

## Our idea
CartGuard compares **what the camera sees each person put in the basket** with **what that person was billed for**, and flags the difference for review.

1. **Detect products and people.** A custom-trained YOLOv8n model recognises 200 retail product classes. A stock YOLOv8n model detects people. Both run with tracking.
2. **Build a vision basket.** A *basket zone* is drawn on the camera view. A product counts only while its centre is inside the zone. Counts are median-smoothed over about 25 frames, and each track's class is stabilised by voting, so tracker ID switches and class flicker do not inflate the bill.
3. **Infer pickup and put-back.** When a class count rises the system logs a `PICKUP`. When it falls it logs a `PUT_BACK`.
4. **Link products to people.** Each product is linked to the nearest person. The link is stabilised by voting over recent frames, and the person ID is attached to every pickup and put-back. This gives a separate basket and bill per person.
5. **Reconcile.** Each person's vision basket is compared item by item and by total with their checkout bill. Under-billed items (seen but not billed) and over-billed items (billed but not seen) are reported with their rupee value.

```
Camera -> YOLOv8 product model ─┐
                                ├-> basket zone + smoothing -> pickup / put-back events
Camera -> YOLOv8 person model ──┘          |                          |
                                    link product -> person    per-person baskets
                                                                       |
                                              compare with each person's bill -> alert
```

## How product-to-person linking works
- A product is linked to a person if it is within `RADIUS` pixels of that person's box. If it is inside the box the distance is 0, and ties are broken by distance to the box centre.
- Each product track keeps its last 15 nearest-person votes. The owner is the majority, and only if it has at least 60% of the votes. Otherwise the owner is `None` (unsure), and those items are surfaced for manual review.
- A pickup is credited to the person most often near the item that entered the zone. A put-back is taken from whoever holds that item, or from the nearest person to the basket if several do.
- Yellow lines on the live view connect each product to its linked person, and the per-person running bill is shown on screen.


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
| Model | 3.4M parameters, about 5.4 ms/image inference |
| Training | 30 epochs, 640 px, batch 8, about 1.2 h on an RTX 3050 6 GB laptop GPU |

These are detector metrics on the dataset's validation images, **not** end-to-end cart accuracy.

**Reconciliation output.** Example from a recorded test run, showing the format of the alert for one person:

```
--- person 2728 ---
product: 162_seasoner x1 at checkout but not detected by camera
product: 165_personal_hygiene x1 added to cart but not present at checkout
product: 173_personal_hygiene x3 added to cart but not present at checkout
product: 180_tissue x1 at checkout but not detected by camera
product: 48_instant_noodles x1 added to cart but not present at checkout
product: 7_puffed_food x1 added to cart but not present at checkout
camera predicted higher bill than checkout by 360 ...
```

## Repository layout
```
prototype.ipynb      # training, live basket + person linking, reconciliation
models/best.pt       # trained product detector weights
demo/                # screenshots used in this README
requirements.txt
```

## Setup and run
```bash
pip install -r requirements.txt
```
1. Put the trained weights at `models/best.pt` and make sure the notebook loads them from there.
2. Open `prototype.ipynb` and run the cells in order. Training needs the dataset at `grocery_datset/data.yaml`. Skip the training cell to use the provided weights.
3. In the live cell, a window opens on the first camera frame. **Drag a box around the basket and press ENTER.** Place items in the basket. The console prints `PICKUP` and `PUT_BACK` events with the person ID, and the window shows each person's running bill.
4. Press **c** to finish (the result is stored as `machine_output`), or **q** to quit.
5. In the checkout cell, set the person ID to the one shown on screen and enter that person's bill, then run it. Use `print(list(machine_output["people"]))` to see which IDs the camera linked items to.

To use a recorded clip instead of the webcam, replace `cv2.VideoCapture(0)` with `cv2.VideoCapture("clip.mp4")`.

## Known limitations
- **Noisy event log.** The pickup/put-back log flickers when detections briefly drop in and out, so the same item can be logged as picked up and put back several times within seconds (visible in the saved notebook output). The smoothed final basket is steadier than the raw event stream.
- **Person IDs are not stable.** The tracker can assign a new ID to the same person (for example after they leave the frame or are occluded), which can split one person's items across several IDs. In the demo the checkout bill is matched by the ID shown on screen. In a real deployment the app user would be bound to the tracked person at the entrance.
- **One fixed basket zone and a fixed camera.** `RADIUS` and the zone are tuned by hand for our camera view.
- **Overlapping people can swap ownership** of an item.
- **The mobile cart is simulated** with a dictionary, and prices come from a demo catalogue table.
- Items stacked on top of each other can hide one another and be undercounted.
- No end-to-end accuracy study yet, only a small number of manual tests.

## Roadmap
- Tie products to the hand holding them with pose keypoints (wrists) for more reliable linking and pickup/put-back.
- Debounce the event log and make person identities persistent.
- Connect to a real mobile-app cart API and run an alert and verification flow.
- Evaluate end to end on recorded shopping sessions.

## Dataset and licence
[- Dataset: TODO (name, source link, licence).](https://universe.roboflow.com/groceries-jxjfd/grocery-goods

Provided by a Roboflow user
License: Public Domain)
- This project uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), which is licensed under AGPL-3.0. Choose this repository's licence accordingly. Licence: TODO.
