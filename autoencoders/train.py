import torch
import torch.nn as nn

def train_autoencoder(model, epochs, criterion, data_generator, position_encoder, grid_size, encoder_optimizer, encoder_scheduler, decoder_optimizer, decoder_scheduler, device):
    lr_list = []
    loss_list = []
    
    model = model.to(device)

    for epoch in range(epochs):
        model.train()

        raw_points = data_generator(grid_size, randomize=True).to(device)
        train_data = position_encoder(raw_points).to(device)
        reconstructed_data = model(train_data)
        loss = criterion(reconstructed_data, raw_points)

        lr_list.append((encoder_scheduler.get_last_lr()[0], decoder_scheduler.get_last_lr()[0]))

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        encoder_scheduler.step()
        decoder_scheduler.step()

        # Calculate loss separately with a new grid
        with torch.no_grad():
            model.eval()
            raw_points = data_generator(grid_size, randomize=False).to(device)
            test_data = position_encoder(raw_points).to(device)
            reconstructed_test_data = model(test_data)
            test_loss = nn.MSELoss()(reconstructed_test_data, raw_points)
            loss_list.append(test_loss.item())

        if (epoch + 1) % (epochs // 10) == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss_list[-1]:.5f}")

    return model, lr_list, loss_list

