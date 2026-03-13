1. set loss is sum  
       loss = torch.nn.MSELoss(reduction='sum')(pred, y_batch) * 100
2. set loss is mean
3. add the train numbers ,batchsize and epoch
4. change learning rate from 2e-5 to 1e-3
5. expand learning ratio * 10
6. expand learning ration * 100 and add accuracy
7. unet with convtranspose
8. unet with convtranspose + interpolat
