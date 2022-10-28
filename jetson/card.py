from dataclasses import dataclass, field
from typing import Dict
import jetson.utils
import jetson.inference
import numpy as np

class DetectedObject:
    pass

@dataclass
class Card:
    numpyImg: np.ndarray
    cudaImg: jetson.utils.cudaImage
    offsetX: int
    offsetY: int
    detectedObjects: Dict[str, jetson.inference.detectNet.Detection] = field(default_factory=dict)

    def add_object(self, name, detection):
        detection.Left += self.offsetX
        detection.Right += self.offsetX
        detection.Top += self.offsetY
        detection.Bottom += self.offsetY
        
        self.detectedObjects[name] = detection

    @staticmethod
    def from_numpy(numpyImg, x, y):
        return Card(
            numpyImg = numpyImg,
            offsetX = x,
            offsetY = y,
            cudaImg = jetson.utils.cudaFromNumpy(numpyImg),
        )

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

def find_match(cards):
    if len(cards) < 2 or cards is None:
        return None, 0.0, []
    matches = []
    for i, card1 in enumerate(cards[:-1]):
        for card2 in cards[i+1:]:
            _matches = intersection(card1.detectedObjects.keys(), card2.detectedObjects.keys())
            matches += [(x, np.mean([card1.detectedObjects[x].Confidence, card2.detectedObjects[x].Confidence]), [card1.detectedObjects[x], card2.detectedObjects[x]]) for x in _matches]
    print(matches)
    if len(matches) == 0:
        return None, 0.0, []
    return sorted(matches, key = lambda x: x[1])[-1]
