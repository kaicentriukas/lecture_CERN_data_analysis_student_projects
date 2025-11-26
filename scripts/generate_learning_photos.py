import os
import random
import matplotlib.pyplot as plt

# Create output folder
OUTPUT_DIR = "../dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pool of math formulas (you can expand later)
FORMULAS = [
    r"x^2 + 2x + 1",
    r"\sqrt{x}",
    r"\frac{1}{x}",
    r"\frac{a}{b}",
    r"\frac{dy}{dx}",
    r"\int_0^1 x^2 dx",
    r"\int e^x dx",
    r"\sum_{n=1}^{\infty} \frac{1}{n^2}",
    r"\sum_{i=0}^{n} i^2",
    r"\lim_{x\to 0} \frac{\sin x}{x}",
    r"\alpha + \beta = \gamma",
    r"\cos(\theta) = \frac{adj}{hyp}",
    r"\sin^2(x) + \cos^2(x) = 1",
    r"e^{i\pi} + 1 = 0",
    r"ax^2 + bx + c = 0",
    r"\ln(x)",
    r"\log_{10}(x)",
    r"\vec{v} \cdot \vec{u}",
    r"\binom{n}{k}",
    r"\sqrt{a^2 + b^2}",
    r"\frac{n!}{k!(n-k)!}",
    r"\int \frac{1}{x} dx",
    r"\frac{d}{dx}(x^n) = nx^{n-1}",
    r"\phi = \frac{1+\sqrt{5}}{2}",
    r"\int_0^\pi \sin x \, dx",
    r"\sum_{n=0}^\infty x^n",
    r"\nabla \cdot \vec{E} = \frac{\rho}{\epsilon_0}",
    r"f(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}",
    r"\Gamma(n) = (n-1)!",
    r"\partial_t u = D \nabla^2 u",
]

# Generate 100 formulas total
NUM_IMAGES = 100
FORMULA_LIST = random.choices(FORMULAS, k=NUM_IMAGES)

def save_formula_image(formula, index):
    plt.figure(figsize=(4, 1), dpi=150)
    plt.text(0.5, 0.5, f"${formula}$", fontsize=22, ha='center', va='center')
    plt.axis('off')
    
    filepath = os.path.join(OUTPUT_DIR, f"formula_{index:03d}.png")
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0.2)
    plt.close()

# Generate and save images
for i, f in enumerate(FORMULA_LIST):
    save_formula_image(f, i)

print(f"Done! {NUM_IMAGES} images saved to '{OUTPUT_DIR}/'")
