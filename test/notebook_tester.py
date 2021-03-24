import click
import glob
import os
import subprocess

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


TEMP_PREFIX = "TEMP_"


@click.command()
@click.argument("filename_or_dirname")
def main(filename_or_dirname):
    if filename_or_dirname.endswith(".ipynb"):
        filenames = [filename_or_dirname]
    else:
        filenames = glob.glob(filename_or_dirname + "*.ipynb")

    for filename in filenames:
        path, basename = os.path.split(filename)
        output_filename = TEMP_PREFIX + basename
        output_filename = os.path.join(path, output_filename)
        cmd = [
            'jupyter', 'nbconvert',
            '--to', 'python',
            filename,
            '--stdout']
        proc = subprocess.run(cmd, stdout=subprocess.PIPE)
        b = proc.stdout
        contents = b.decode('utf8')
        contents = contents.replace("get_ipython()", "# get_ipython()")
        with open(output_filename, "wt") as f:
            f.write(contents)
        print("Running {}".format(output_filename))
        try:
            subprocess.run(
                ['python', output_filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True)
        except subprocess.CalledProcessError as err:
            print(err)
        else:
            print("Completed {} with no errors".format(output_filename))
        finally:
            os.remove(output_filename)

if __name__ == '__main__':
    main()
