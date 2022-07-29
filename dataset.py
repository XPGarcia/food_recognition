from tensorflow import keras
import cv2
import os
import numpy as np


# classes for data loading and preprocessing
class Dataset:
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
                                                (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
                                                (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['water',
               'pear',
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
               'gluten-free-bread',
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
               'bread',
               'strawberries',
               'wine-rosa-c',
               'watermelon-fresh',
               'green-asparagus',
               'white-asparagus',
               'peach']

    def __init__(
        self,
        images_dir,
        masks_dir,
        classes=None,
        augmentation=None,
        preprocessing=None,
        segment=0
     ):
        self.ids = os.listdir(images_dir)
        # segmentation
        segment_size = 1000
        if len(self.ids) < (segment+1)*segment_size:
            self.ids = self.ids[segment * segment_size:]
        else:
            self.ids = self.ids[segment * segment_size:(segment + 1) * segment_size]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # COLOR_BGR2RGB
        r, g, b = cv2.split(image)
        image = cv2.merge((r, g, b))

        # extract certain classes from mask
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
