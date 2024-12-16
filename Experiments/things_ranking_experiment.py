from psychopy import visual, core, event, gui, data
import os, random, csv
import pandas as pd
import numpy as np

def ensure_directory_exists(filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

def initialize_dataframe(images, filename):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        return pd.read_csv(filename)
    else:
        # Create a new DataFrame with just the image names
        df = pd.DataFrame({'Image Name': [os.path.basename(img) for img in images]})
        df.to_csv(filename, index=False)  # Save the empty DataFrame with headers
        return df

def update_and_save_dataframe(df, filename, participant_name, rankings, phase):
    phase_col = f'Participant {participant_name} {phase} Ordering'
    if phase_col not in df.columns:
        df[phase_col] = pd.NA  # Initialize new columns for new participant

    for img, rank in rankings.items():
        img_name = os.path.basename(img)
        df.loc[df['Image Name'] == img_name, phase_col] = float(rank)

    df.to_csv(filename, index=False)
    print("Results saved.")

def perform_ranking(win, images, prompt, layout):
    image_stimuli = []
    rankings = {}
    rank = 1

    for i, img in enumerate(images):
        pos_x, pos_y = layout[i]
        image_stim = visual.ImageStim(win, image=img, pos=(pos_x, pos_y), size=(0.15, 0.15))
        image_stimuli.append(image_stim)

    instruction = visual.TextStim(win, text=prompt, pos=(0, 0.9), height=0.04)
    mouse = event.Mouse(win=win)
    escape_pressed = False

    while image_stimuli:
        win.flip(clearBuffer=True)
        for img_stim in image_stimuli:
            img_stim.draw()
        instruction.draw()

        if mouse.getPressed()[0]:
            for img_stim in image_stimuli[:]:
                if img_stim.contains(mouse):
                    rankings[img_stim.image] = rank
                    image_stimuli.remove(img_stim)
                    rank += 1
                    core.wait(0.2)
                    break

        keys = event.getKeys()
        if 'escape' in keys:
            escape_pressed = True
            break

    return rankings, escape_pressed

def default_layout(n):
    positions = []
    for i in range(n):
        row = i // 10
        col = i % 10
        pos_x = -0.9 + 0.2 * col
        pos_y = 0.4 - 0.2 * row
        positions.append((pos_x, pos_y))
    return positions

def calculate_average_rankings(rankings1, rankings2, images):
    # Calculate the average ranking for each image
    average_rankings = {}
    for img in images:
        img_name = os.path.basename(img)
        rank1 = rankings1.get(img, np.nan)  # Use np.nan for images not ranked due to early termination
        rank2 = rankings2.get(img, np.nan)
        if np.isnan(rank1) or np.isnan(rank2):
            average_rank = np.nanmean([rank1, rank2])  # Calculate mean ignoring NaN
        else:
            average_rank = (rank1 + rank2) / 2
        average_rankings[img] = average_rank
    
    # Sort images by average rankings
    sorted_images = sorted(average_rankings, key=average_rankings.get)
    return sorted_images

info = {'Participant Name': ''}
infoDlg = gui.DlgFromDict(dictionary=info, title='Image Complexity Ranking')
if not infoDlg.OK:
    core.quit()

image_folder = '/home/wallacelab/complexity-final/Images/THINGSFiltered'
images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(images)
images = images[:50]  # Ensure exactly 50 images

win = visual.Window([1920, 1080], fullscr=True, allowGUI=False, color='black')
filename = 'output/file/name'
ensure_directory_exists(filename)
df = initialize_dataframe(images, filename)

# Adjustments within the try block:
try:
    rankings1, escape_pressed = perform_ranking(win, images, "Click on the most complex image. It will disappear after you click it. Then, click on the next most complex image, and the next one, until the last image left is the least complex image. Wait a second after you click this last image for the next stage to load.", default_layout(50))
    update_and_save_dataframe(df, filename, info["Participant Name"], rankings1, "First")
    if escape_pressed:
        win.close()
        core.quit()

    random.shuffle(images)
    rankings2, escape_pressed = perform_ranking(win, images, "Click again on the most complex image. It will disappear after you click it. Then, click on the next most complex image, and the next one, until the last image left is the least complex image. Wait a second after you click this last image for the next stage to load.", default_layout(50))
    update_and_save_dataframe(df, filename, info["Participant Name"], rankings2, "Second")
    if escape_pressed:
        win.close()
        core.quit()

    # Calculate average rankings and sort images for the third step
    sorted_images = calculate_average_rankings(rankings1, rankings2, images)
    rankings3, escape_pressed = perform_ranking(win, sorted_images, "The order of these images from left to right represents your previous responses from most complex in the top left corner to least complex in the bottom right corner. Once again, click on the most complex image until there are no more images left. This is your opportunity to change the order of your responses. At the end, you will be able to see how your response compared to other people's.", default_layout(len(sorted_images)))
    update_and_save_dataframe(df, filename, info["Participant Name"], rankings3, "Third")
    if escape_pressed:
        win.close()
        core.quit()

finally:
    win.close()
    core.quit()