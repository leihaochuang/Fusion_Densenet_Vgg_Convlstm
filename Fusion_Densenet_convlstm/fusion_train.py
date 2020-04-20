import torch
import fusion_config
import time
from fusion_config import args_setting
from fusion_dataset import RoadSequenceDatasetList, RoadSequenceDataset
from fusion_model import generate_model
from torchvision import transforms
from torch.optim import lr_scheduler



def train(args, epoch, model, train_loader, device, optimizer, criterion):
    since = time.time()
    model.train()
    for batch_idx,  sample_batched in enumerate(train_loader):
        # LongTensor
        if args.model == 'Densenet-Vgg':
            data, data2, target = sample_batched['data'].to(device), \
                                 sample_batched['data2'].to(device), \
                                 sample_batched['label'].type(torch.LongTensor).to(device)

            data = {
                'image': data, 'point_cloud_image': data2
            }
        else:
            data,  target = sample_batched['data'].to(device), \
                                  sample_batched['label'].type(torch.LongTensor).to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, int(batch_idx * len(data)), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    time_elapsed = time.time() - since
    print('Train Epoch: {} complete in {:.0f}m {:.0f}s'.format(epoch,
        time_elapsed // 60, time_elapsed % 60))


def val(args, model, val_loader, device, criterion, best_acc):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample_batched in val_loader:
            if args.model == 'Densenet-Vgg':
                data, data2, target = sample_batched['data'].to(device), \
                                      sample_batched['data2'].to(device), \
                                      sample_batched['label'].type(torch.LongTensor).to(device)

                data = {
                    'image': data, 'point_cloud_image': data2
                }
            else:
                data, target = sample_batched['data'].to(device), \
                               sample_batched['label'].type(torch.LongTensor).to(device)
            output = model(data)

            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= (len(val_loader.dataset)/args.test_batch_size)
    val_acc = 100. * int(correct) / (len(val_loader.dataset) * fusion_config.label_height * fusion_config.label_width)
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)\n'.format(
        test_loss, int(correct), len(val_loader.dataset), val_acc))
    torch.save(model.state_dict(), '%s.pth'%val_acc)


def get_parameters(model, layer_name):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.UpsamplingBilinear2d
    )
    for name, module in model.named_children():
        if name in layer_name:
            for layer in module.children():
                if isinstance(layer, modules_skipped):
                    continue
                else:
                    for parma in layer.parameters():
                        yield parma


if __name__ == '__main__':

    start = time.time()
    args = args_setting()
    torch.manual_seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # turn image into floatTensor
    op_tranforms = transforms.Compose([transforms.ToTensor()])

    # load data for batches, num_workers for multiprocess
    if args.model == 'Densenet-Vgg':
        train_loader = torch.utils.data.DataLoader(
            RoadSequenceDatasetList(file_path=fusion_config.image_train_path,
                                    file_path_Point_cloud_image=fusion_config.point_cloud_image_train_path,
                                    transforms=op_tranforms),
            batch_size=args.batch_size, shuffle=True, num_workers=fusion_config.data_loader_numworkers)
        val_loader = torch.utils.data.DataLoader(
            RoadSequenceDatasetList(file_path=fusion_config.image_train_path,
                                    file_path_Point_cloud_image=fusion_config.point_cloud_image_train_path,
                                    transforms=op_tranforms),
            batch_size=args.test_batch_size, shuffle=True, num_workers=fusion_config.data_loader_numworkers)
    elif args.model == 'Densenet':
        train_loader = torch.utils.data.DataLoader(
            RoadSequenceDataset(file_path=fusion_config.image_train_path,
                                    transforms=op_tranforms),
            batch_size=args.batch_size, shuffle=True, num_workers=fusion_config.data_loader_numworkers)
        val_loader = torch.utils.data.DataLoader(
            RoadSequenceDataset(file_path=fusion_config.image_train_path,
                                    transforms=op_tranforms),
            batch_size=args.test_batch_size, shuffle=True, num_workers=fusion_config.data_loader_numworkers)

    # load model
    model = generate_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    class_weight = torch.Tensor(fusion_config.class_weight)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)
    best_acc = 0
    model_dict = model.state_dict()

    # train
    for epoch in range(1, args.epochs+1):
        train(args, epoch, model, train_loader, device, optimizer, criterion)
        val(args, model, val_loader, device, criterion, best_acc)
        scheduler.step()
    end = time.time()
    print(f"训练总耗时为：{(end-start)//60}m")
