## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from models import Encoder, Decoder

from datasets import VoxelSDFDataset

# latent dim is the size of the latent dimension z, conv_dim is how many convolution layers there are on the first conv, decoder_dim is an array with network
def train(latent_dim, conv_dim, decoder_dim):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f'DEVICE: {device}')
    data = VoxelSDFDataset()

    training_set, validation_set = torch.utils.data.random_split(data, [int(0.7 * len(data)), len(data) - int(0.7 * len(data))])

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=8, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=8, shuffle=False)
    
    VoxelEncoder = Encoder(1, conv_dim, latent_dim)
    VoxelDecoder = Decoder(latent_dim, decoder_dim)
    
    VoxelEncoder.to(device)
    VoxelDecoder.to(device)

    encoder_optimizer = optim.Adam(VoxelEncoder.parameters(), lr=0.0001)
    decoder_optimizer = optim.Adam(VoxelDecoder.parameters(), lr=0.0001)
    
    def train_one_epoch(epoch_index, clamp_delta=0.2):
        avg_loss = 0
        count = 0
        for idx, data in enumerate(training_loader):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            voxel_data, [grid, near, surf] = data
            
            loss = 0
            
            for i, voxel in enumerate(voxel_data):
                vox_obj = voxel[None, None,:,:,:].float() # [1, 1, 64, 64, 64]
                latents = VoxelEncoder(vox_obj)
                
                def computeLoss(points, sdfs):
                    l = 0
                    for i in range(len(points)):
                        p = points[None, i,]
                        sdf = sdfs[i,]
                        z = torch.cat((latents, p), dim=1)
                        sdf_hat = VoxelDecoder(z.float())

                        l += abs(torch.clamp(sdf_hat - sdf, min=-clamp_delta, max=clamp_delta))
                    return l

                loss += computeLoss(grid[0][i,], grid[1][i,])
                loss += computeLoss(near[0][i,], near[1][i,])
                loss += computeLoss(surf[0][i,], surf[1][i,])

            
            loss.backward()
            
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            
            # AUX STUFF
            if idx % 100 == 0:
                print(f'loss at {idx}: {idx}')
                torch.save(VoxelEncoder, f'./save/encoder_l{latent_dim}_c{conv_dim}_e{epoch_index}.pt')
                torch.save(VoxelDecoder, f'./save/decoder_l{latent_dim}_c{conv_dim}_e{epoch_index}.pt')
            
            avg_loss += loss
            count += 1
        return avg_loss / count

    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        VoxelEncoder.train(True)
        VoxelDecoder.train(True)
        training_loss = train_one_epoch(epoch_number, clamp_delta=0.25)
        print(f'TRAINING LOSS: {training_loss}')

        # We don't need gradients on to do reporting
        VoxelEncoder.train(False)
        VoxelDecoder.train(False)
        
        def val_one_epoch(epoch_index, clamp_delta=0.2):
            avg_loss = 0
            count = 0
            for i, data in enumerate(validation_loader):
                
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                voxel_data, [grid, near, surf] = data
                
                loss = 0
                
                for i, voxel in enumerate(voxel_data):
                    vox_obj = voxel[None, None,:,:,:].float() # [1, 1, 64, 64, 64]
                    latents = VoxelEncoder(vox_obj)
                    
                    def computeLoss(points, sdfs):
                        l = 0
                        for i in range(len(points)):
                            p = points[None, i,]
                            sdf = sdfs[i,]
                            z = torch.cat((latents, p), dim=1)
                            sdf_hat = VoxelDecoder(z.float())

                            l += abs(torch.clamp(sdf_hat - sdf, min=-clamp_delta, max=clamp_delta))
                        return l

                    loss += computeLoss(grid[0][i,], grid[1][i,])
                    loss += computeLoss(near[0][i,], near[1][i,])
                    loss += computeLoss(surf[0][i,], surf[1][i,])

                avg_loss += loss
                count += 1
            
            return avg_loss / count
        
        if epoch % 10 == 0:
            val_loss = val_one_epoch(epoch_number, clamp_delta=0.25)
            print(f'VALIDATION LOSS: {val_loss}')

def main():
    train(16, 3, [16,32,32,64])

if __name__ == '__main__':
    main()

