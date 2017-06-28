#! /usr/bin/python

import click
import yaml

@click.command()
@click.argument('key')
@click.argument('run_id')
@click.argument('output_id')
def main(key, run_id, output_id):

    with open('floyd/ids.yml', 'r') as f:
        # use safe_load instead load
        runs = yaml.safe_load(f)

    runs[key]["run"] = str(run_id)
    runs[key]["output"] = str(output_id)

    with open('floyd/ids.yml', 'w') as f:
        # use safe_load instead load
        yaml.dump(runs, f, default_flow_style=False)


if __name__ == '__main__':
    main()
