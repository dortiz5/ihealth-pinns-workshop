# Importaciones de Matplotlib
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# Definir el colormap personalizado
# Definir los colores en RGB
rgb = {'red': ((0.0, 0.0, 0.0),
                (0.5, 1.0, 1.0),
                (1.0, 1.0, 1.0)),

        'green': ((0.0, 0.0, 0.0),
                    (0.5, 1.0, 1.0),
                    (1.0, 0.0, 0.0)),

        'blue': ((0.0, 1.0, 1.0),
                (0.5, 1.0, 1.0),
                (1.0, 0.0, 0.0))
        }
# Definir el colormap
cmap_ = LinearSegmentedColormap('RedGreenBlue', rgb)

# Actualización de los parámetros de Matplotlib
gray = '#5c5c5c' #'#5c5c5c' '000'
mpl.rcParams.update(
    {
        "text.color" : gray,
        "xtick.color" :gray,
        "ytick.color" :gray,
        "axes.labelcolor" : gray,
        "axes.edgecolor" :gray,
        "axes.spines.right" : False,
        "axes.spines.top" : False,
        'font.size' : 13,
        'figure.constrained_layout.use': True,
        'interactive': False,
        "font.family": 'serif',  # Use the Computer modern font
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
    }
)

mpl.rcParams.update(
    {
        'text.usetex': False,
        'mathtext.fontset': 'stix',
    }
)