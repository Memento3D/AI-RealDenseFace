// Rebuild the offline viewer using the frontend's existing esbuild installation.
const path=require('path');
const fs=require('fs');
const [frontendRoot,outputFile]=process.argv.slice(2);
if(!frontendRoot||!outputFile)throw new Error('Usage: node build_viewer.cjs <frontend-root> <new-output.js>');
const output=path.resolve(outputFile);
if(fs.existsSync(output))throw new Error('Use a fresh output path; existing review artifacts are preserved.');
const esbuild=require(require.resolve('esbuild',{paths:[path.resolve(frontendRoot)]}));
esbuild.buildSync({entryPoints:[path.join(__dirname,'viewer.js')],bundle:true,minify:true,format:'esm',platform:'browser',target:'es2022',legalComments:'inline',outfile:output});
console.log(output);
