def print_losses(epoch, mean_loss, mean_l_patch, mean_l_patch_dx, mean_l_patch_dy):
    # Print to console
    print('-------------------')
    print('---- END of EPOCH  ')
    print("---- ", end='')
    print("loss: {:10.7f}, ".format(mean_loss), end='')
    print("l_patch: {:10.7f}, ".format(mean_l_patch), end='')
    print("l_patch_dx: {:10.7f}, ".format(mean_l_patch_dx), end='')
    print("l_patch_dy: {:10.7f}, ".format(mean_l_patch_dy), end='')
    print("   ")

    print('-------------- TRAINING OF EPOCH ' + str(0 + epoch + 1).zfill(2) + 'FINISH ---------------')
    print('---------------------------------------------------------')
    print('   ')
    print('   ')
    print('   ')