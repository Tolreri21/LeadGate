# LeadGate demo — local serving vs live Lambda, one raw lead in.
Output "demo.gif"

Set Shell "bash"
Set FontSize 18
Set Width 1240
Set Height 780
Set Padding 26
Set Theme "Dracula"
Set TypingSpeed 28ms

# ---------- hidden setup: PATH, creds, URL, cache warm-up (never recorded) ----------
Hide
Type `export PATH=/Users/anatolijperederij/.local/bin:/opt/anaconda3/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin`
Enter
Type `cd /Users/anatolijperederij/PycharmProjects/LeadGate`
Enter
Type `export AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id)`
Enter
Type `export AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key)`
Enter
Type `export FUNCTION_URL=https://5wvjvavku7z45evzcqgmdvshrm0qrocu.lambda-url.us-east-1.on.aws/`
Enter
Type `export PS1='$ '`
Enter
Type `uv run python -c 'import json,leadgate.serving as s; m,t=s.load_artifacts(); s.predict_lead(m,t,json.load(open("lead.json")))' >/dev/null 2>&1`
Enter
Type `curl -s --aws-sigv4 'aws:amz:us-east-1:lambda' --user "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" -H 'content-type: application/json' -X POST "$FUNCTION_URL" --data @lead.json >/dev/null`
Enter
Type `clear`
Enter
Show

Sleep 600ms

# ---------- the request contract ----------
Type `# LeadGate: raw lead in -> call decision out.  (no 'duration' field — that's the whole point)`
Enter
Sleep 900ms
Type `cat lead.json | jq -c`
Enter
Sleep 2500ms

# ---------- [1/2] local serving, no AWS ----------
Type `# [1/2] LOCAL  — model.joblib straight from ./models, zero AWS`
Enter
Sleep 700ms
Type `uv run python -c 'import json,leadgate.serving as s; m,t=s.load_artifacts(); print(json.dumps(s.predict_lead(m,t,json.load(open("lead.json"))),indent=2))'`
Enter
Sleep 3200ms

# ---------- [2/2] live deployed Lambda, SigV4-signed ----------
Type `# [2/2] LIVE   — SigV4-signed to the deployed Lambda Function URL (us-east-1)`
Enter
Sleep 700ms
Type `curl -s --aws-sigv4 'aws:amz:us-east-1:lambda' --user "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" -H 'content-type: application/json' -X POST "$FUNCTION_URL" --data @lead.json | jq`
Enter
Sleep 3500ms

Type `# same model, same answer:  local == prod  OK`
Enter
Sleep 2600ms
