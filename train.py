import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from flickr8k_loader import Flickr8kDataset
from model import DecoderCNN, EncoderCNN


image_size = (224, 224)
batch_size = 256
shuffle = True
num_workers = 8
learning_rate = 0.001
num_epochs = 100

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size = image_size), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

flickr8k = Flickr8kDataset(root_dir = '/home/kartik/data/flickr8k', transform = transform)
print(len(flickr8k))
# result = flickr8k[0]
# print(result['name'], result['image'].shape, result['captions'].shape)
print(flickr8k.get_vocab_size())

dataloader = DataLoader(dataset = flickr8k, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
encoder = EncoderCNN()
encoder = encoder.to(device = device)
# encoder = encoder.cuda()
decoder = DecoderCNN(growth_rate = 3, num_layers = 5, target_size = flickr8k.max_sequence_length)
decoder = decoder.to(device = device)
# decoder = decoder.cuda()
# for parameter in decoder.parameters():
#     parameter.type(torch.cuda.FloatTensor)
for parameter in decoder.parameters():
    print(parameter.type())

criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr = 0.001)

for epoch in range(num_epochs):
    print('-' * 50)
    print('Epoch', epoch)
    print('-' * 50)
    for i, sample in enumerate(dataloader):
        images = sample['image']
        # print('Batch:', i, 'Image name:', sample['name'], 'Captions shape:', sample['captions'].shape)
        features = encoder(images)
        # print(features.size())
        outputs = decoder(features)
        # print(outputs.size())
        sample['captions'] = sample['captions']
        for j in range(sample['captions'].shape[1]): # Number of captions per image
            targets = sample['captions'][:, j, :].long()
            # print('Caption iteration:', j, 'Target size:', target.size())
            loss = criterion(outputs, torch.max(targets, 1)[1])
            decoder.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()

        print('Loss', loss.item())
        if(i % 5) == 0:
            torch.save(decoder.state_dict(), 'models/decoder_batch_' + str(batch_size) + '_epoch' + str(epoch) + '_iteration_' + str(i) + '.pt')
    torch.save(decoder.state_dict(), 'models/decoder_batch_' + str(batch_size) + '_epoch' + str(epoch) + '.pt')
