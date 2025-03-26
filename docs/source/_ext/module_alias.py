# This extension will rewrite the modules vmmath and inp_out (at the moment) with their alias
import os
import re

def modify_rst_before_build(app, config):
    rst_dir = os.path.join(app.confdir, 'api')
    
    # Mapping of actual module names to desired documentation names
    module_aliases = {
        'pyLOM.vmmath': 'pyLOM.math',
        'pyLOM.inp_out': 'pyLOM.io'
    }

    try:
        for filename in os.listdir(rst_dir):
            if filename.endswith('.rst'):
                filepath = os.path.join(rst_dir, filename)
                
                # Check if this file needs to be renamed
                for original, alias in module_aliases.items():
                    if filename == f"{original}.rst":
                        # Rename the file
                        new_filepath = os.path.join(rst_dir, f"{alias}.rst")
                        os.rename(filepath, new_filepath)
                        filepath = new_filepath
                        break
                
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Replace module names in the content
                for original, alias in module_aliases.items():
                    # Replace module name in automodule directive
                    content = re.sub(
                        rf'(.. automodule:: ){original}(\n)',
                        fr'\1{alias}\2',
                        content
                    )
                    
                    # Replace all occurrences of the original module name
                    content = content.replace(original, alias)
                    
                    # Replace module name in headings and other references
                    content = content.replace(
                        f"{original.split('.')[-1]}",
                        f"{alias.split('.')[-1]}"
                    )
                    content = content.replace(
                        f"module {original}", 
                        f"module {alias}"
                    )
                
                with open(filepath, 'w') as f:
                    f.write(content)
                
    except Exception as e:
        print(f"Error modifying RST files: {e}")

def modify_html_files(app, exception):
    if exception is not None:
        return

    # Mapping of actual module names to desired documentation names
    module_aliases = {
        'pyLOM.vmmath': 'pyLOM.math',
        'pyLOM.inp_out': 'pyLOM.io'
    }

    # Directory where HTML files are generated
    html_dir = os.path.join(app.outdir)

    try:
        for root, dirs, files in os.walk(html_dir):
            for filename in files:
                if filename.endswith('.html'):
                    filepath = os.path.join(root, filename)
                    
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    modified = False
                    
                    # Replace module names in the content
                    for original, alias in module_aliases.items():
                        if original in content:
                            content = content.replace(original, alias)
                            modified = True
                    
                    # Write back if modified
                    if modified:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
    except Exception as e:
        print(f"Error modifying HTML files: {e}")

def setup(app):
    # Connect to the config-inited event to modify RST files before build
    app.connect('config-inited', modify_rst_before_build)
    
    # Connect to the build-finished event to modify HTML files
    app.connect('build-finished', modify_html_files)
    
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }