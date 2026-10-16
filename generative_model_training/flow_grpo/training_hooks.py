"""Schedule checkpoint saves using completed optimizer steps."""


def run_training_hooks(config, global_step, save, *, training_finished=False):
    if config.debug:
        return
    if training_finished:
        if global_step > 0 and (config.save_freq <= 0 or global_step % config.save_freq != 0):
            save()
        return
    if global_step <= 0:
        return
    if config.save_freq > 0 and global_step % config.save_freq == 0:
        save()
