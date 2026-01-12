import { NextRequest, NextResponse } from 'next/server';
import { execFile } from 'child_process';
import path from 'path';

export async function POST(req: NextRequest) {
    try {
        const body = await req.json();

        // Path to the python script
        // Assuming we are in project root, script is in tacrolimus-service/predict_cli.py
        const scriptPath = path.join(process.cwd(), 'tacrolimus-service', 'predict_cli.py');

        // We need to pass the body as a JSON string argument
        const inputJson = JSON.stringify(body);

        return new Promise((resolve) => {
            // Adjust 'python' to 'python3' or specific path if needed. 
            // On Windows, 'python' usually works if in PATH.
            execFile('python', [scriptPath, '--input', inputJson], (error, stdout, stderr) => {
                if (error) {
                    console.error('Python execution error:', error);
                    console.error('Stderr:', stderr);
                    resolve(NextResponse.json(
                        { error: 'Failed to execute prediction model', details: stderr },
                        { status: 500 }
                    ));
                    return;
                }

                try {
                    const result = JSON.parse(stdout);
                    if (result.status === 'error') {
                        resolve(NextResponse.json(
                            { error: result.error },
                            { status: 400 }
                        ));
                    } else {
                        resolve(NextResponse.json(result));
                    }
                } catch (parseError) {
                    console.error('JSON parse error:', parseError);
                    resolve(NextResponse.json(
                        { error: 'Invalid response from model', details: stdout },
                        { status: 500 }
                    ));
                }
            });
        });

    } catch (e) {
        return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
    }
}
