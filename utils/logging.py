

def log_train_step(loss, loss_student, loss_recons, example_ct, epoch):
    print(f'Iter: {example_ct} of Epoch: {epoch} | Total Loss:{loss:.3f}, ST:{loss_student:.3f}, Recons:{loss_recons:.3f}')


def log_accuracy(correct, neg, pos):
    print()
    print('=======================================')
    print(f'\nValidation KNN Accuracy: {correct} - Negatives: {neg} - Positives: {pos}')
    print('=======================================')