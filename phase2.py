import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance ,ImageFilter

image_path = "0.jpg"
grid_size = 4  # 4x4
# grid_size = 2  #2*2 

img = cv2.imread(image_path)

def enhance_image(img):
    img = img.filter(ImageFilter.MedianFilter(3))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)
    return img

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape
piece_h, piece_w = h // grid_size, w // grid_size


pieces = []
for r in range(grid_size):
    for c in range(grid_size):
        y1, y2 = r*piece_h, (r+1)*piece_h
        x1, x2 = c*piece_w, (c+1)*piece_w
        pieces.append(img[y1:y2, x1:x2])

num_pieces = len(pieces)


def get_edges(piece):
    return {
        'top': piece[0,:,:].astype(float),
        'bottom': piece[-1,:,:].astype(float),
        'left': piece[:,0,:].astype(float),
        'right': piece[:,-1,:].astype(float)
    }

edges = [get_edges(p) for p in pieces]


def ssd(edge1, edge2):
    diff = edge1 - edge2
    return np.sum(diff**2)


compatibility = np.full((num_pieces, num_pieces, 4), np.inf)
for i in range(num_pieces):
    for j in range(num_pieces):
        if i == j:
            continue
        compatibility[i,j,0] = ssd(edges[i]['bottom'], edges[j]['top'])  
        compatibility[i,j,1] = ssd(edges[i]['top'], edges[j]['bottom'])  
        compatibility[i,j,2] = ssd(edges[i]['right'], edges[j]['left'])  
        compatibility[i,j,3] = ssd(edges[i]['left'], edges[j]['right'])   


best_grid = None
best_score = np.inf

def backtrack(pos, grid, used, current_score):
    global best_grid, best_score

    if current_score >= best_score:
        return  

    if pos == num_pieces:
      
        best_grid = [row[:] for row in grid]
        best_score = current_score
        return

    r, c = divmod(pos, grid_size)

    for i in range(num_pieces):
        if used[i]:
            continue

       
        score = 0
        if r > 0:
            top_idx = grid[r-1][c]
            score += compatibility[top_idx, i, 0]
        if c > 0:
            left_idx = grid[r][c-1]
            score += compatibility[left_idx, i, 2]

        
        grid[r][c] = i
        used[i] = True

        
        backtrack(pos+1, grid, used, current_score + score)

        
        grid[r][c] = None
        used[i] = False


initial_grid = [[None]*grid_size for _ in range(grid_size)]
used = [False]*num_pieces
backtrack(0, initial_grid, used, 0)


reconstructed = np.zeros_like(img)
for r in range(grid_size):
    for c in range(grid_size):
        idx = best_grid[r][c]
        y1, y2 = r*piece_h, (r+1)*piece_h
        x1, x2 = c*piece_w, (c+1)*piece_w
        reconstructed[y1:y2, x1:x2] = pieces[idx]


reconstructed_pil = Image.fromarray(reconstructed)
enhancer = ImageEnhance.Contrast(reconstructed_pil)
reconstructed_pil = enhancer.enhance(1.5)
enhancer = ImageEnhance.Sharpness(reconstructed_pil)
reconstructed_pil = enhancer.enhance(2.0)


plt.figure(figsize=(8,8))
plt.imshow(reconstructed)
plt.axis('off')
plt.show()
