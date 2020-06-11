'''
Developer's name: ABDELRAHMAN S. ABDELRAHMAN

Date: 10th of June 2020

Purpose:
    training the model

'''

import torch


def training_network(model, optimizer, criterion, trainloader, validloader, epochs=20, processor='gpu'):
    """
    This function trains the model

    Parameters:
     model, optimizer, epochs, criterion, trainloader, processor='gpu'

    Returns:
        None

    """
    if processor == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'
    steps = 0
    running_loss = 0
    print_every = 40
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in trainloader:
            steps += 1
            # GPU
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        output = model.forward(images)
                        valid_loss += criterion(output, labels).item()

                        # Acuuracy Calculations
                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader):.3f}")

                running_loss = 0
                model.train()
    print("___________Congratulation The model is trained Successfully___________")
