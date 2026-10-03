# 👟 Shoe Type Classification using Deep Learning

Classifies a shoe photo into **Sneakers, Boots, Sandals, Heels or Loafers** using
**MobileNetV2 (transfer learning)** with TensorFlow/Keras. Includes a **Streamlit** web app
where you upload an image, click **Predict**, and see the shoe type + confidence.
Runs on a normal Windows laptop CPU. No GPU needed.

---

## Project structure
```
ShoeTypeClassification/
├── app.py                 <- Streamlit web app
├── train_model.ipynb      <- Training notebook (run this first)
├── requirements.txt       <- Python libraries
├── README.md
├── model/                 <- shoe_classifier.keras is created here after training
├── raw_images/            <- downloaded / original images (per class)
├── dataset/
│   ├── train/{sneakers,boots,sandals,heels,loafers}
│   └── validation/{sneakers,boots,sandals,heels,loafers}
└── sample_images/         <- put test photos here for the demo
```
> The trained model is **not** in the ZIP (it must be trained on your laptop). Step 4 does that.

---

## Step-by-step setup (Windows + Anaconda)

### Step 1: Install Anaconda
Download from https://www.anaconda.com/download and install it (keep default options).

### Step 2: Unzip the project
Unzip `ShoeTypeClassification.zip`, for example to `C:\Projects\ShoeTypeClassification`.

### Step 3: Create an environment and install libraries
Open **Anaconda Prompt** (Start menu) and run these one at a time:

```
cd C:\Projects\ShoeTypeClassification
conda create -n shoe_env python=3.10 -y
conda activate shoe_env
pip install -r requirements.txt
python -m ipykernel install --user --name shoe_env --display-name "Python (shoe_env)"
```
(Python 3.10 is used because it works smoothly with TensorFlow 2.15. Installing takes a few minutes.)

### Step 4: Train the model (once)
In the same Anaconda Prompt:
```
jupyter notebook
```
1. Chrome opens Jupyter. Click **train_model.ipynb**.
2. Menu **Kernel -> Change kernel -> Python (shoe_env)**.
3. Run cells one by one with **Shift + Enter**, top to bottom.
4. Step 3 downloads the images (needs internet, about 5-10 minutes).
5. Training takes about 10-20 minutes on CPU.
6. At the end, `model/shoe_classifier.keras` is created. 🎉

### Step 5: Run the web app
Close the notebook (or open a second Anaconda Prompt), then:
```
cd C:\Projects\ShoeTypeClassification
conda activate shoe_env
streamlit run app.py
```
Chrome opens at **http://localhost:8501**. If it does not, type that address in Chrome.

### Step 6: Use it
1. Click **Browse files** and choose a shoe image.
2. Click **Predict**.
3. See the shoe type, confidence %, and a probability bar for every class.

To stop the app: press `Ctrl + C` in the Anaconda Prompt.

---

## Using your own dataset instead (optional)
1. Download a shoe dataset (e.g. from Kaggle, search "shoe classification dataset").
2. Copy images into `raw_images/sneakers`, `raw_images/boots`, `raw_images/sandals`, `raw_images/heels`, `raw_images/loafers`.
3. In notebook Step 3 set `DOWNLOAD_IMAGES = False`, then run the remaining cells.

## How it works (for your viva / presentation)
1. **Dataset**: shoe images in 5 folders, split 80% train / 20% validation.
2. **Preprocessing**: resize to 224x224, scale pixels to [-1, 1].
3. **Augmentation**: random flip, rotation, zoom, contrast to reduce overfitting.
4. **Transfer learning**: MobileNetV2 pretrained on ImageNet extracts features; we add GlobalAveragePooling + Dropout + Dense(5, softmax).
5. **Training**: Stage 1 trains only the new layer; Stage 2 fine-tunes the last 30 layers with a tiny learning rate.
6. **Evaluation**: accuracy, classification report, confusion matrix.
7. **Deployment**: Streamlit loads the `.keras` model, preprocesses the upload, and shows the prediction + confidence (softmax probability).

## Troubleshooting
| Problem | Fix |
|---|---|
| `streamlit` is not recognized | Run `conda activate shoe_env` first |
| "Model file not found" in the app | Finish training (notebook Step 10) first |
| Image download gives few or 0 images | Check internet, rerun the cell, or use the manual dataset option |
| Low accuracy | Delete wrong images in `raw_images`, increase `IMAGES_PER_CLASS` to 300, rerun from Step 4 |
| Training too slow | Lower `EPOCHS` to 5 and `FINE_TUNE_EPOCHS` to 2 |
| Kernel not found in Jupyter | Re-run the `ipykernel install` command from Step 3 |
| Out of memory | Set `BATCH_SIZE = 16` |

## Demo tips
* Train the model **before** the demo and keep `model/shoe_classifier.keras` safe.
* Keep 5-6 clear test photos in `sample_images/` (plain background works best).
* Start the app a few minutes early; the first load takes a few seconds.

## Possible improvements (future work)
Add more classes (flip-flops, slippers), use a larger dataset, deploy online (Streamlit Cloud), convert to TensorFlow Lite for mobile.
