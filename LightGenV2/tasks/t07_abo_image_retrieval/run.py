"""T07 independent entry point. Run from this folder: python run.py --help."""
if __package__:
    from .standalone.cli import main
else:
    from standalone.cli import main

if __name__ == '__main__':
    main()
