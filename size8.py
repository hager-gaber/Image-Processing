import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "0.jpg"
grid_size = 8

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w, _ = img.shape
piece_h, piece_w = h // grid_size, w // grid_size

pieces = []
for r in range(grid_size):
    for c in range(grid_size):
        pieces.append(
            img[r*piece_h:(r+1)*piece_h, c*piece_w:(c+1)*piece_w]
        )

num_pieces = len(pieces)

def get_edges(piece):
    return {
        'top': piece[0,:,:].astype(float),
        'bottom': piece[-1,:,:].astype(float),
        'left': piece[:,0,:].astype(float),
        'right': piece[:,-1,:].astype(float)
    }

edges = [get_edges(p) for p in pieces]

def ssd(e1, e2):
    return np.sum((e1 - e2)**2)


grid = [[None]*grid_size for _ in range(grid_size)]
used = [False]*num_pieces

grid[0][0] = 0
used[0] = True

for r in range(grid_size):
    for c in range(grid_size):
        if r == 0 and c == 0:
            continue

        best_idx = None
        best_score = np.inf

        for i in range(num_pieces):
            if used[i]:
                continue

            score = 0
            if r > 0:
                score += ssd(edges[grid[r-1][c]]['bottom'], edges[i]['top'])
            if c > 0:
                score += ssd(edges[grid[r][c-1]]['right'], edges[i]['left'])

            if score < best_score:
                best_score = score
                best_idx = i

        grid[r][c] = best_idx
        used[best_idx] = True


reconstructed = np.zeros_like(img)
for r in range(grid_size):
    for c in range(grid_size):
        idx = grid[r][c]
        reconstructed[
            r*piece_h:(r+1)*piece_h,
            c*piece_w:(c+1)*piece_w
        ] = pieces[idx]


plt.figure(figsize=(10,10))
plt.imshow(reconstructed)
plt.axis('off')
plt.show()
plt.figure(figsize=(10,10))
plt.imshow(reconstructed)

for r in range(grid_size):
    for c in range(grid_size):
        y = r*piece_h + piece_h//2
        x = c*piece_w + piece_w//2
        plt.text(x, y, str(grid[r][c]), color='yellow', fontsize=8)

plt.title("Greedy Edge Matching Result")
plt.axis('off')
plt.show()