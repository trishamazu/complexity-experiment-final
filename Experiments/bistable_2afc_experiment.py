from psychopy import visual, core, event, gui, data
import os, random, csv
import pandas as pd

def load_image_pairs(folder):
    valid_extensions = ('.png', '.jpg', '.jpeg')  # Add other image formats as needed
    images = [file for file in os.listdir(folder) if file.lower().endswith(valid_extensions) and not file.startswith('.')]
    images.sort()  # Sort to ensure order if filesystem doesn't
    pairs = [(images[i], images[i+1]) for i in range(0, len(images) - 1, 2)]
    return pairs

def save_results(results, filename='output/file/name'):
    df = pd.DataFrame(results)
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        # Load existing data
        existing_df = pd.read_csv(filename)
        
        # Ensure that all columns from new data exist in existing data, if not add them
        for column in df.columns:
            if column not in existing_df.columns:
                existing_df[column] = None  # Initialize as None or an appropriate default value
        
        # Concatenate new data to existing data
        combined_df = pd.concat([existing_df, df], sort=False).reset_index(drop=True)
        
        # Save combined data back to CSV
        combined_df.to_csv(filename, index=False)

info = {'Participant Number': ''}
infoDlg = gui.DlgFromDict(dictionary=info, title='Image Complexity Rating')
if not infoDlg.OK:
    core.quit()

participant_num = info["Participant Number"]
response_col = f'Participant {participant_num} Choice'
time_col = f'Participant {participant_num} Response Time'

image_folder = '/home/wallacelab/complexity-final/Images/Bistable-Control'
image_pairs = load_image_pairs(image_folder)
image_pairs *= 5  # Repeat each pair 5 times
random.shuffle(image_pairs)  # Shuffle pairs

win = visual.Window([800, 600], fullscr=True, allowGUI=False)
results = {'First Image': [], 'Second Image': [], response_col: [], time_col: []}

try:
    for first_image, second_image in image_pairs:
        random_order = random.sample([first_image, second_image], 2)  # Shuffle images within each pair
        left_image = visual.ImageStim(win, image=os.path.join(image_folder, random_order[0]), pos=(-0.4, 0), size=(0.5, 0.5))
        right_image = visual.ImageStim(win, image=os.path.join(image_folder, random_order[1]), pos=(0.4, 0), size=(0.5, 0.5))
        instruction = visual.TextStim(win, text="Which image is more complex? Use the left and right arrow keys to choose.", pos=(0, 0.8))
        
        win.flip()
        event.clearEvents()
        timer = core.Clock()  # Start timer
        escape_pressed = False
        while True:
            left_image.draw()
            right_image.draw()
            instruction.draw()
            win.flip()

            keys = event.getKeys(keyList=['left', 'right', 'escape'], timeStamped=timer)
            if keys:
                key, time = keys[0]  # Get the first key and its timestamp
                if key == 'escape':
                    escape_pressed = True
                    break
                elif key in ['left', 'right']:
                    chosen_image = random_order[0] if key == 'left' else random_order[1]
                    results['First Image'].append(random_order[0])
                    results['Second Image'].append(random_order[1])
                    results[response_col].append(chosen_image)
                    results[time_col].append(time)
                    break
        if escape_pressed:
            break
finally:
    save_results(results)
    win.close()
    core.quit()