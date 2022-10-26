# Errors
## RuntimeError: expected scalar type Double but found Float
My data was in float64 but nn.funcional.linear takes number with float32 type

## Boolean value of Tensor with more than one value is ambiguous
When defining a loss function, you have to instantiate first. 
Ex) criterion = nn.MSELoss()
    loss = criterion(output, target)

## Default process group has not been initialized, please make sure to call init_process_group.
I tried to use distributed data parallel with my CPU. Usually we need GPU to do it.