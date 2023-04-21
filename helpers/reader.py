from dataset import ImageClassificationDataset
import torchvision.transforms as transforms
import json

def read_dataset():
    f = open('config.json')
    json_data = json.load(f)
    TRANSFORMS = transforms.Compose([
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    datasets = {}
    for name in json_data["config"]["datasets"]:
        datasets[name] = ImageClassificationDataset(json_data["config"]["task"] + '_' + name,  json_data["config"]["labels"], TRANSFORMS)

    dataset = datasets[json_data["config"]["datasets"][0]]
    
    print(json_data)
    print(json_data["config"]["task"])
    print("{0} task with {1} categories defined".format(json_data["config"]["task"], json_data["config"]["labels"]))
    
    categories = json_data["config"]["labels"]
    
    return (datasets, dataset)