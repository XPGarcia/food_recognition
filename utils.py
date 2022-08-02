import numpy as np
import matplotlib.pyplot as plt
import albumentations as A


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)  # , cmap='binary')
    plt.show()


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=224, min_width=224, always_apply=True, border_mode=0),
        # A.RandomCrop(height=224, width=224, always_apply=True),

        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),

        A.OneOf(
            [
                # A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=8),
                # A.RandomGamma(p=1),
            ],
            p=1.9,
        ),

        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(224, 224)
    ]
    return A.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

def get_classes():
    return ['pear',
               'water',
               'egg',
               'grapes',
               'butter',
               'bread-white',
               'jam',
               'bread-whole-wheat',
               'apple',
               'tea-green',
               'white-coffee-with-caffeine',
               'tea-black',
               'mixed-salad-chopped-without-sauce',
               'cheese',
               'tomato-sauce',
               'pasta-spaghetti',
               'carrot',
               'onion',
               'beef-cut-into-stripes-only-meat',
               'rice-noodles-vermicelli',
               'salad-leaf-salad-green',
               'bread-grain',
               'espresso-with-caffeine',
               'banana',
               'mixed-vegetables',
               'bread-wholemeal',
               'savoury-puff-pastry',
               'wine-white',
               'dried-meat',
               'fresh-cheese',
               'red-radish',
               'hard-cheese',
               'ham-raw',
               'bread-fruit',
               'oil-vinegar-salad-dressing',
               'tomato',
               'cauliflower',
               'potato-gnocchi',
               'wine-red',
               'sauce-cream',
               'pasta-linguini-parpadelle-tagliatelle',
               'french-beans',
               'almonds',
               'dark-chocolate',
               'mandarine',
               'semi-hard-cheese',
               'croissant',
               'sushi',
               'berries',
               'biscuits',
               'thickened-cream-35',
               'corn',
               'celeriac',
               'alfa-sprouts',
               'chickpeas',
               'leaf-spinach',
               'rice',
               'chocolate-cookies',
               'pineapple',
               'tart',
               'coffee-with-caffeine',
               'focaccia',
               'pizza-with-vegetables-baked',
               'soup-vegetable',
               'bread-toast',
               'potatoes-steamed',
               'spaetzle',
               'frying-sausage',
               'lasagne-meat-prepared',
               'boisson-au-glucose-50g',
               'ma1-4esli',
               'peanut-butter',
               'chips-french-fries',
               'mushroom',
               'ratatouille',
               'veggie-burger',
               'country-fries',
               'yaourt-yahourt-yogourt-ou-yoghourt-natural',
               'hummus',
               'fish',
               'beer',
               'peanut',
               'pizza-margherita-baked',
               'pickle',
               'ham-cooked',
               'cake-chocolate',
               'bread-french-white-flour',
               'sauce-mushroom',
               'rice-basmati',
               'soup-of-lentils-dahl-dhal',
               'pumpkin',
               'witloof-chicory',
               'vegetable-au-gratin-baked',
               'balsamic-salad-dressing',
               'pasta-penne',
               'tea-peppermint',
               'soup-pumpkin',
               'quiche-with-cheese-baked-with-puff-pastry',
               'mango',
               'green-bean-steamed-without-addition-of-salt',
               'cucumber',
               'bread-half-white',
               'pasta',
               'beef-filet',
               'pasta-twist',
               'pasta-wholemeal',
               'walnut',
               'soft-cheese',
               'salmon-smoked',
               'sweet-pepper',
               'sauce-soya',
               'chicken-breast',
               'rice-whole-grain',
               'bread-nut',
               'green-olives',
               'roll-of-half-white-or-white-flour-with-large-void',
               'parmesan',
               'cappuccino',
               'flakes-oat',
               'mayonnaise',
               'chicken',
               'cheese-for-raclette',
               'orange',
               'goat-cheese-soft',
               'tuna',
               'tomme',
               'apple-pie',
               'rosti',
               'broccoli',
               'beans-kidney',
               'white-cabbage',
               'ketchup',
               'salt-cake-vegetables-filled',
               'pistachio',
               'feta',
               'salmon',
               'avocado',
               'sauce-pesto',
               'salad-rocket',
               'pizza-with-ham-baked',
               'gruya-re',
               'ristretto-with-caffeine',
               'risotto-without-cheese-cooked',
               'crunch-ma1-4esli',
               'braided-white-loaf',
               'peas',
               'chicken-curry-cream-coconut-milk-curry-spices-paste',
               'bolognaise-sauce',
               'bacon-frying',
               'salami',
               'lentils',
               'mushrooms',
               'mashed-potatoes-prepared-with-full-fat-milk-with-butter',
               'fennel',
               'chocolate-mousse',
               'corn-crisps',
               'sweet-potato',
               'bircherma1-4esli-prepared-no-sugar-added',
               'beetroot-steamed-without-addition-of-salt',
               'sauce-savoury',
               'leek',
               'milk',
               'tea',
               'fruit-salad',
               'bread-rye',
               'salad-lambs-ear',
               'potatoes-au-gratin-dauphinois-prepared',
               'red-cabbage',
               'praline',
               'bread-black',
               'black-olives',
               'mozzarella',
               'bacon-cooking',
               'pomegranate',
               'hamburger-bread-meat-ketchup',
               'curry-vegetarian',
               'honey',
               'juice-orange',
               'cookies',
               'mixed-nuts',
               'breadcrumbs-unspiced',
               'chicken-leg',
               'raspberries',
               'beef-sirloin-steak',
               'salad-dressing',
               'shrimp-prawn-large',
               'sour-cream',
               'greek-salad',
               'sauce-roast',
               'zucchini',
               'greek-yaourt-yahourt-yogourt-ou-yoghourt',
               'cashew-nut',
               'meat-terrine-pata-c',
               'chicken-cut-into-stripes-only-meat',
               'couscous',
               'bread-wholemeal-toast',
               'craape-plain',
               'bread-5-grain',
               'tofu',
               'water-mineral',
               'ham-croissant',
               'juice-apple',
               'falafel-balls',
               'egg-scrambled-prepared',
               'brioche',
               'bread-pita',
               'pasta-haprnli',
               'blue-mould-cheese',
               'vegetable-mix-peas-and-carrots',
               'quinoa',
               'crisps',
               'beef',
               'butter-spread-puree-almond',
               'beef-minced-only-meat',
               'hazelnut-chocolate-spread-nutella-ovomaltine-caotina',
               'chocolate',
               'nectarine',
               'ice-tea',
               'applesauce-unsweetened-canned',
               'syrup-diluted-ready-to-drink',
               'sugar-melon',
               'bread-sourdough',
               'rusk-wholemeal',
               'bread',
               'shrimp-prawn-small',
               'french-salad-dressing',
               'pancakes',
               'milk-chocolate',
               'pork',
               'dairy-ice-cream',
               'guacamole',
               'sausage',
               'herbal-tea',
               'fruit-coulis',
               'water-with-lemon-juice',
               'brownie',
               'lemon',
               'veal-sausage',
               'dates',
               'roll-with-pieces-of-chocolate',
               'taboula-c-prepared-with-couscous',
               'croissant-with-chocolate-filling',
               'eggplant',
               'sesame-seeds',
               'cottage-cheese',
               'fruit-tart',
               'cream-cheese',
               'tea-verveine',
               'tiramisu',
               'grits-polenta-maize-flour',
               'pasta-noodles',
               'artichoke',
               'blueberries',
               'mixed-seeds',
               'caprese-salad-tomato-mozzarella',
               'omelette-plain',
               'hazelnut',
               'kiwi',
               'dried-raisins',
               'kolhrabi',
               'plums',
               'beetroot-raw',
               'cream',
               'fajita-bread-only',
               'apricots',
               'kefir-drink',
               'gluten-free-bread',
               'strawberries',
               'wine-rosa-c',
               'watermelon-fresh',
               'green-asparagus',
               'white-asparagus',
               'peach']
