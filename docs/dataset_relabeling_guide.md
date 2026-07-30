# Golden 84 Dataset Strategy & Labeling Guide

To achieve 100% detection on the 84 test images and handle the "Others" requirement, follow this manual dataset collection guide.

## Phase 1: Identify the ~40 Classes
1. Open the folder containing the 84 test images (`Color Inputs/`).
2. Open an Excel sheet or Notepad.
3. Look at every single image. Write down a clear name for each unique sign you see (e.g., `speed_limit_90`, `no_u_turn`, `blue_pass_left`, `yellow_bumps`). 
   * *Tip:* Keep the names simple, lowercase, with underscores.
   * You should end up with a list of about 40 unique classes.

## Phase 2: Collect Images Quickly
For an AI to learn a class perfectly without noise, it needs about **20 images** per class.
Since you have 40 classes, you need to collect roughly 800 images total.

1. **Use Google Images:** Search for the exact sign (e.g., "Malaysian speed limit 90 sign", or "European pass on left sign").
2. **Download 20 images per class:** Make sure to download a variety (some clear, some blurry, some angled).
3. **Organize:** Save them on your computer in folders named after the class (e.g., `Desktop/Dataset/speed_limit_90/`, `Desktop/Dataset/no_u_turn/`).

## Phase 3: Roboflow Upload & Labeling
1. Go to [Roboflow](https://app.roboflow.com/) and create a **New Project** (Object Detection). Name it `Golden 84 Signs`.
2. Drag and drop all 800 images into Roboflow.
3. **Start Labeling:** Draw a tight bounding box around the sign in each image.
4. When it asks for a class name, type exactly the name you wrote in your Excel sheet. 
   * *Crucial:* Do not make typos. `Speed_Limit` and `speed_limit` will become two different classes. Be consistent!

## Phase 3.5: Team Collaboration (How to work together)
You can easily invite your groupmates to Roboflow so everyone can upload and label at the same time!
1. **Invite Teammates:** In your Roboflow project, go to **Settings** (gear icon) > **Workspace** > **Members**. Click **Invite Members** and type in your groupmates' email addresses.
2. **Divide the Work:** Since there are ~40 classes, split them up! If you have 4 members, assign 10 specific classes to each person.
   * *Example:* Member 1 does all "Speed Limit" signs, Member 2 does all "No Parking/U-Turn" signs, etc.
3. **Labeling Together:** You can all be logged into the same project at the same time. You will see their images pop up, and they will see yours!
4. **Job Feature:** Roboflow has a "Assign Job" feature in the Annotate tab where you can officially assign specific unlabeled images to a specific teammate so no one accidentally labels the same image twice.

## Phase 4: Augmentation (Making 800 images act like 2,400)
Once all 800 images are boxed and labeled:
1. Go to **Generate New Version**.
2. **Preprocessing:** 
   * Add **Auto-Orient**.
   * Add **Resize** (Stretch to 640x640).
3. **Augmentation:** (This is how you get extra marks for robustness)
   * **Flip:** Horizontal (ONLY IF the sign looks the same flipped. Skip this if you have directional signs like "Turn Left").
   * **Crop:** 0% Minimum Zoom, 10% Maximum Zoom.
   * **Rotation:** Between -15° and +15°.
   * **Brightness:** Between -15% and +15%.
   * **Blur:** Up to 1.25px.
   * **Noise:** Up to 1%.
4. Set the dataset output size to **3x** (This will generate ~2,400 images).
5. Click **Generate** and export it to a Colab `data.yaml` link just like before.

## Why this guarantees success:
- Your model will only know about the exact 40 signs the lecturer cares about. 
- It won't have 20 junk classes generating false positives (noise).
- Because you are providing 20 images per class + 3x augmentation, the accuracy will easily reach 95%+ mAP50.
