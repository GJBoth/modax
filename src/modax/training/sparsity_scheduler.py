from dataclasses import dataclass


@dataclass
class mask_scheduler:
    patience: int = 500
    delta: float = 1e-5
    periodicity: int = 200

    periodic: bool = False
    best_loss = None
    best_iteration = None

    def __call__(self, loss, iteration, optimizer):
        if self.periodic is True:
            if (iteration - self.best_iteration) % self.periodicity == 0:
                update_mask, optimizer = True, optimizer
            else:
                update_mask, optimizer = False, optimizer

        elif self.best_loss is None:
            self.best_loss = loss
            self.best_iteration = iteration
            self.best_optim_state = optimizer
            update_mask, optimizer = False, optimizer

        # If it didnt improve, check if we're past patience
        elif (self.best_loss - loss) < self.delta:
            if (iteration - self.best_iteration) >= self.patience:
                self.periodic = True  # switch to periodic regime
                self.best_iteration = iteration  # because the iterator doesnt reset
                update_mask, optimizer = True, self.best_optim_state
            else:
                update_mask, optimizer = False, optimizer

        # If not, keep going
        else:
            self.best_loss = loss
            self.best_iteration = iteration
            self.best_optim_state = optimizer
            update_mask, optimizer = False, optimizer

        return update_mask, optimizer
