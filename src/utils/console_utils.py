from rich import print


def print_error(message, verbose=True):
    if verbose:
        print(f"[bold][red][ERROR][/red][/bold] {message}")


def print_warning(message, verbose=True):
    if verbose:
        print(f"[bold][yellow][WARNING][/yellow][/bold] {message}")


def print_info(message, verbose=True):
    if verbose:
        print(f"[bold][blue][INFO][/blue][/bold] {message}")


