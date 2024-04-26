from dataloader.cifar_dataloader import *
from models.ViT import VisionTransformer
from models.prune import unstructured_prune_model
from scripts.train import Trainer, Trainer_LRP, Trainer_Prune
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import os

torch.backends.cudnn.benchmark = True
os.environ["WANDB_RUN_GROUP"] = "SHVN"

def cifar_10_vit_base():
    # Hyper parameters
    EPOCHS = 300
    BATCH_SIZE = 1024
    
    NUM_CLASSES = 10
    PATCH_SIZE = 4
    IMG_SIZE = 32
    IN_CHANNELS = 3
    NUM_HEADS = 8
    DROPOUT = 0.1
    HIDDEN_DIM = 512
    DEPTH = 8
    LAMBDA = 0.1
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMBED_DIM = 128
    NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
    
    train_dataloader, test_dataloader = get_SHVN_dataloader(batch_size=BATCH_SIZE, num_workers=20)
    
    STEPS_PER_EPOCH = len(train_dataloader)
    
    model = VisionTransformer(image_size=IMG_SIZE, 
                              patch_size=PATCH_SIZE, 
                              num_patches=NUM_PATCHES, 
                              in_channels=IN_CHANNELS, 
                              embed_dim=EMBED_DIM, 
                              num_classes=NUM_CLASSES, 
                              depth=DEPTH, 
                              heads=NUM_HEADS, 
                              mlp_dim=HIDDEN_DIM, 
                              device=DEVICE,
                              dropout=DROPOUT,
                              lrp=False).to(DEVICE)
    
    summary(model, (3,32,32))
    
    torch.save(model.state_dict(), "./model_weights/ViT_SHVN_initial_model_weights.pt")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, anneal_strategy='cos')
    
    trainer = Trainer(model, NUM_CLASSES, optimizer, criterion, scheduler, wandb_log=True, project_name="EML-Final-Project", experiment_name="ViT_SHVN")
    
    trainer.train(train_dataloader, test_dataloader, EPOCHS)
    
def cifar_10_vit_lrp():
    # Hyper parameters
    EPOCHS = 300
    BATCH_SIZE = 1024
    
    NUM_CLASSES = 10
    PATCH_SIZE = 4
    IMG_SIZE = 32
    IN_CHANNELS = 3
    NUM_HEADS = 8
    DROPOUT = 0.1
    HIDDEN_DIM = 512
    DEPTH = 8
    LRP = True
    LAMBDAS = [0.1, 0.05, 0.01, 0.005]
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMBED_DIM = 128
    NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
    
    train_dataloader, test_dataloader = get_SHVN_dataloader(batch_size=BATCH_SIZE, num_workers=20)
    
    STEPS_PER_EPOCH = len(train_dataloader)
    
    for LAMBDA in LAMBDAS:
        model = VisionTransformer(image_size=IMG_SIZE, 
                                patch_size=PATCH_SIZE, 
                                num_patches=NUM_PATCHES, 
                                in_channels=IN_CHANNELS, 
                                embed_dim=EMBED_DIM, 
                                num_classes=NUM_CLASSES, 
                                depth=DEPTH, 
                                heads=NUM_HEADS, 
                                mlp_dim=HIDDEN_DIM, 
                                device=DEVICE,
                                dropout=DROPOUT,
                                lrp=True).to(DEVICE)
        
        #summary(model, (3,32,32))
        
        #torch.save(model.state_dict(), "./model_weights/ViT_CIFAR_10_initial_model_weights.pt")
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, anneal_strategy='cos')
        
        trainer = Trainer_LRP(model=model, 
                            num_classes=NUM_CLASSES, 
                            optimizer=optimizer, 
                            criterion=criterion, 
                            scheduler=scheduler, 
                            wandb_log=True, 
                            project_name="EML-Final-Project", 
                            experiment_name=f"ViT_SHVN_LRP_{LAMBDA}", 
                            lamb=LAMBDA, 
                            disable_lrp_loss=False)
        
        trainer.train(train_dataloader, test_dataloader, EPOCHS)
        
        lrp_weights =  model.get_lrp_weights()
        
        total = 0
        pruned_8 = 0
        pruned_9 = 0
        for i, weight in enumerate(lrp_weights):
            w = weight.squeeze(0).squeeze(0).abs().detach().cpu().numpy()

            total += len((w.flatten()))
            pruned_8 += np.where(w < 1e-8, 1, 0).sum()
            pruned_9 += np.where(w < 1e-9, 1, 0).sum()

            fig = plt.figure()
            
            plt.matshow(w)
            plt.colorbar()
            plt.xlabel("Embedding Dim")
            plt.ylabel("Head Dim")
            plt.savefig(f"./lrp_weight_viz/SHVN_{i}_lrp_{LAMBDA}.png")
            plt.close()
        
        print(f"[LOG] For Lambda = {LAMBDA}, the pruned number of heads = {pruned_8} giving a pruning ratio of {pruned_8*100/total}%")
        print("\n")
        print(f"[LOG] For Lambda = {LAMBDA}, the pruned number of heads = {pruned_9} giving a pruning ratio of {pruned_9*100/total}%")
        
def cifar_10_vit_unstructured_pruning():
    # Hyper parameters
    EPOCHS = 300
    BATCH_SIZE = 1024
    
    NUM_CLASSES = 10
    PATCH_SIZE = 4
    IMG_SIZE = 32
    IN_CHANNELS = 3
    NUM_HEADS = 8
    DROPOUT = 0.1
    HIDDEN_DIM = 512
    DEPTH = 8
    LAMBDA = 0.1
    LRP = False
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMBED_DIM = 128
    NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
    
    train_dataloader, test_dataloader = get_SHVN_dataloader(batch_size=BATCH_SIZE, num_workers=20)
    
    STEPS_PER_EPOCH = len(train_dataloader)
    
    prune_ratios = [0.3, 0.5, 0.7, 0.9]
    
    for prune_ratio in prune_ratios:
        model = VisionTransformer(image_size=IMG_SIZE, 
                              patch_size=PATCH_SIZE, 
                              num_patches=NUM_PATCHES, 
                              in_channels=IN_CHANNELS, 
                              embed_dim=EMBED_DIM, 
                              num_classes=NUM_CLASSES, 
                              depth=DEPTH, 
                              heads=NUM_HEADS, 
                              mlp_dim=HIDDEN_DIM, 
                              device=DEVICE,
                              dropout=DROPOUT,
                              lrp=LRP).to(DEVICE)
        
        model.load_state_dict(torch.load("./model_weights/ViT_SHVN_epoch_300.pt"))
        
        model = unstructured_prune_model(model, prune_ratio)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, anneal_strategy='cos')
        
        trainer = Trainer_Prune(model, 
                                NUM_CLASSES, 
                                optimizer, 
                                criterion, 
                                scheduler, 
                                wandb_log=True, 
                                project_name="EML-Final-Project", 
                                experiment_name=f"ViT_SHVN_Prune_{prune_ratio}")
        
        trainer.train(train_dataloader, test_dataloader, EPOCHS, save_every=None)

def cifar_10_vit_lottery_ticket():
    # Hyper parameters
    EPOCHS = 300
    BATCH_SIZE = 1024
    
    NUM_CLASSES = 10
    PATCH_SIZE = 4
    IMG_SIZE = 32
    IN_CHANNELS = 3
    NUM_HEADS = 8
    DROPOUT = 0.1
    HIDDEN_DIM = 512
    DEPTH = 8
    LAMBDA = 0.1
    LRP = False
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMBED_DIM = 128
    NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
    
    train_dataloader, test_dataloader = get_SHVN_dataloader(batch_size=BATCH_SIZE, num_workers=20)
    
    STEPS_PER_EPOCH = len(train_dataloader)
    
    prune_ratios = [0.3, 0.5, 0.7, 0.9]

    masks = dict()
    
    # Get Prune masks
    for prune_ratio in prune_ratios:
        model = VisionTransformer(image_size=IMG_SIZE, 
                              patch_size=PATCH_SIZE, 
                              num_patches=NUM_PATCHES, 
                              in_channels=IN_CHANNELS, 
                              embed_dim=EMBED_DIM, 
                              num_classes=NUM_CLASSES, 
                              depth=DEPTH, 
                              heads=NUM_HEADS, 
                              mlp_dim=HIDDEN_DIM, 
                              device=DEVICE,
                              dropout=DROPOUT,
                              lrp=LRP).to(DEVICE)
        
        # Load Final Weights
        model.load_state_dict(torch.load("./model_weights/ViT_SHVN_epoch_300.pt"))
        
        # Prune Model
        model = unstructured_prune_model(model, prune_ratio)
        masks[prune_ratio] = dict()

        # Get masks
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                if "mlp_head" not in name:
                    weight_copy = m.weight.data.abs().clone()
                    mask = weight_copy.gt(0).float().cuda()
                    masks[prune_ratio][name] = mask

    for prune_ratio in prune_ratios:
        model = VisionTransformer(image_size=IMG_SIZE, 
                              patch_size=PATCH_SIZE, 
                              num_patches=NUM_PATCHES, 
                              in_channels=IN_CHANNELS, 
                              embed_dim=EMBED_DIM, 
                              num_classes=NUM_CLASSES, 
                              depth=DEPTH, 
                              heads=NUM_HEADS, 
                              mlp_dim=HIDDEN_DIM, 
                              device=DEVICE,
                              dropout=DROPOUT,
                              lrp=LRP).to(DEVICE)
        
        # Load Initial Weights
        model.load_state_dict(torch.load("./model_weights/ViT_SHVN_initial_model_weights.pt"))
        
        # Apply Masks
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                if "mlp_head" not in name:
                    m.weight.data.mul_(masks[prune_ratio][name])
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, anneal_strategy='cos')
        
        trainer = Trainer_Prune(model, 
                                NUM_CLASSES, 
                                optimizer, 
                                criterion, 
                                scheduler, 
                                wandb_log=True, 
                                project_name="EML-Final-Project", 
                                experiment_name=f"ViT_SHVN_LT_Prune_{prune_ratio}"
                                )
        
        trainer.train(train_dataloader, test_dataloader, EPOCHS, save_every=None)

def cifar_10_vit_early_bird():
    epoch_set = [10, 30, 50, 100, 150]
    pruning_ratios = [0.3, 0.5]

    # Hyper parameters
    EPOCHS = 300
    BATCH_SIZE = 1024
    
    NUM_CLASSES = 10
    PATCH_SIZE = 4
    IMG_SIZE = 32
    IN_CHANNELS = 3
    NUM_HEADS = 8
    DROPOUT = 0.1
    HIDDEN_DIM = 512
    DEPTH = 8
    LAMBDA = 0.1
    LRP = False
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMBED_DIM = 128
    NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
    
    train_dataloader, test_dataloader = get_SHVN_dataloader(batch_size=BATCH_SIZE, num_workers=20)
    
    STEPS_PER_EPOCH = len(train_dataloader)

    for epoch in epoch_set:
        for prune_ratio in pruning_ratios:
            model = VisionTransformer(image_size=IMG_SIZE, 
                              patch_size=PATCH_SIZE, 
                              num_patches=NUM_PATCHES, 
                              in_channels=IN_CHANNELS, 
                              embed_dim=EMBED_DIM, 
                              num_classes=NUM_CLASSES, 
                              depth=DEPTH, 
                              heads=NUM_HEADS, 
                              mlp_dim=HIDDEN_DIM, 
                              device=DEVICE,
                              dropout=DROPOUT,
                              lrp=LRP).to(DEVICE)
        
            # Load Epoch Weights Weights
            model.load_state_dict(torch.load(f"./model_weights/ViT_SHVN_epoch_{epoch}.pt"))
            model = unstructured_prune_model(model, prune_ratio)
            mask_ = dict()

            # Get Mask
            for name, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    if "mlp_head" not in name:
                        weight_copy = m.weight.data.abs().clone()
                        mask = weight_copy.gt(0).float().cuda()
                        mask_[name] = mask

            model = VisionTransformer(image_size=IMG_SIZE, 
                              patch_size=PATCH_SIZE, 
                              num_patches=NUM_PATCHES, 
                              in_channels=IN_CHANNELS, 
                              embed_dim=EMBED_DIM, 
                              num_classes=NUM_CLASSES, 
                              depth=DEPTH, 
                              heads=NUM_HEADS, 
                              mlp_dim=HIDDEN_DIM, 
                              device=DEVICE,
                              dropout=DROPOUT,
                              lrp=LRP).to(DEVICE)

            # Load Inital Weights
            model.load_state_dict(torch.load("./model_weights/ViT_SHVN_initial_model_weights.pt"))

            # Apply Masks
            for name, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    if "mlp_head" not in name:
                        m.weight.data.mul_(mask_[name])

            optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                                      max_lr=1e-2, 
                                                      steps_per_epoch=STEPS_PER_EPOCH, 
                                                      epochs=EPOCHS, 
                                                      anneal_strategy='cos')
            
            trainer = Trainer_Prune(model, 
                                    NUM_CLASSES, 
                                    optimizer, 
                                    criterion, 
                                    scheduler, 
                                    wandb_log=True, 
                                    project_name="EML-Final-Project",
                                    experiment_name=f"ViT_SHVN_EB_Prune_{prune_ratio}_Epoch_{epoch}",
                                    )
            
            trainer.train(train_dataloader, test_dataloader, EPOCHS, save_every=None)

if __name__ == "__main__":
    #cifar_10_vit_base()
    #cifar_10_vit_lrp()
    #cifar_10_vit_unstructured_pruning()
    #cifar_10_vit_lottery_ticket()
    cifar_10_vit_early_bird()
