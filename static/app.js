async function getJSON(url){
  const r = await fetch(url, {cache:'no-store'});
  if(!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return await r.json();
}

function fmt(x, d=4){
  if(x === null || x === undefined) return '';
  if(typeof x !== 'number') return x;
  if(!isFinite(x)) return '';
  return x.toFixed(d);
}

function fmtInt(x){
  if(x === null || x === undefined) return '';
  if(typeof x !== 'number') return x;
  return Math.round(x).toString();
}

function renderTable(rows){
  const q = document.getElementById('search').value.trim().toUpperCase();
  const minp = parseFloat(document.getElementById('minp').value || '0');
  const tb = document.querySelector('#tbl tbody');
  tb.innerHTML = '';

  const filtered = rows.filter(r => {
    if(q && !r.symbol.toUpperCase().includes(q)) return false;
    if(typeof r.p_touch_2 === 'number' && r.p_touch_2 < minp) return false;
    return true;
  });

  for(const r of filtered){
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.symbol}</td>
      <td>${fmt(r.price, 6)}</td>
      <td>${fmt(r.p_touch_2, 3)}</td>
      <td>${fmt(r.p_touch_5, 3)}</td>
      <td>${fmt(r.p_touch_10, 3)}</td>
      <td>${fmt(r.dist_to_target_atr_2, 2)}</td>
      <td>${fmt(r.atr_pct, 4)}</td>
      <td>${fmt(r.spread_bps, 1)}</td>
      <td>${fmtInt(r.notional_6h)}</td>
      <td>${fmtInt(r.quote_age_s)}</td>
      <td>${r.updated_utc || ''}</td>
    `;
    tb.appendChild(tr);
  }

  document.getElementById('foot').textContent = `${filtered.length} rows shown (of ${rows.length})`;
}

async function refresh(){
  try{
    const status = await getJSON('/api/status');
    document.getElementById('horizon').textContent = status.config?.horizon_hours ?? '?';
    document.getElementById('interval').textContent = status.config?.scan_interval_minutes ?? '?';
    document.getElementById('status').textContent = JSON.stringify(status, null, 2);
  }catch(e){
    document.getElementById('status').textContent = `Status error: ${e.message}`;
  }

  try{
    const scan = await getJSON('/api/scan');
    renderTable(scan.rows || []);
  }catch(e){
    const tb = document.querySelector('#tbl tbody');
    tb.innerHTML = `<tr><td colspan="11">Scan error: ${e.message}</td></tr>`;
  }
}

document.getElementById('refresh').addEventListener('click', refresh);
document.getElementById('search').addEventListener('input', () => refresh());
document.getElementById('minp').addEventListener('change', () => refresh());

refresh();
setInterval(refresh, 15000);
