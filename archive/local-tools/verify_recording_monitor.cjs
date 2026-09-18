const { chromium } = require('C:/Users/LEGION/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs');
(async()=>{
const browser=await chromium.launch({channel:'msedge',headless:true});
const page=await browser.newPage();
let requests=[];
let state={recording_active:true,current_phase:'recording',dataset_repo_id:'local/test',current_episode:1,total_episodes:5,saved_episodes:0,cameras:['front','wrist'],available_controls:{exit_early:true,rerecord_episode:true,stop_recording:true}};
await page.route('**/*',async route=>{
 const req=route.request(),path=new URL(req.url()).pathname;
 if(path==='/recording-monitor.html')return route.fulfill({contentType:'text/html',body:fs.readFileSync('recording-monitor.html','utf8')});
 if(path==='/recording-status')return route.fulfill({json:state});
 if(path.startsWith('/camera-feed/'))return route.fulfill({contentType:'image/svg+xml',body:'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="#345"/></svg>'});
 if(req.method()==='POST'){requests.push(path);return route.fulfill({json:{success:true}})}
 return route.abort();
});
await page.goto('http://monitor.test/recording-monitor.html');
await page.waitForFunction(()=>document.querySelectorAll('#cameras img').length===2);
await page.locator('#advance').click();
if(requests.join()!=='/recording-exit-early')throw Error('Wrong advance endpoint');
await page.reload();
await page.waitForFunction(()=>document.querySelectorAll('#cameras img').length===2);
if(requests.includes('/start-recording'))throw Error('Unexpected recording start');
state={...state,recording_active:false,current_phase:'completed',available_controls:{}};
await page.waitForFunction(()=>document.getElementById('stop').disabled&&document.querySelectorAll('#cameras img').length===0);
console.log('PASS: two previews, save command, refresh recovery without restarting, inactive controls disabled. All network requests mocked; no hardware accessed.');
await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
