/**
 * Fortune 500 registry + logo helpers.
 * Loads the bundled Fortune 500 CSV first, then falls back to remote datasets
 * and a local seed so the forum/search UI still works during development.
 */
(function () {
  var DATA_URLS = [
    "data/fortune500.csv",
    "https://www.salttechno.ai/datasets/fortune-500-companies-2025.json",
    "https://raw.githubusercontent.com/cmusam/fortune500/master/csv/fortune500-2019.csv"
  ];

  var DOMAIN_OVERRIDES = {
    ...(window.HC_DOMAIN_OVERRIDES || {}),
    "Walmart":"walmart.com","Amazon":"amazon.com","Amazon.com":"amazon.com","UnitedHealth Group":"unitedhealthgroup.com","Apple":"apple.com","CVS Health":"cvshealth.com","Berkshire Hathaway":"berkshirehathaway.com","Alphabet":"abc.xyz","Exxon Mobil":"exxonmobil.com","McKesson":"mckesson.com","Cencora":"cencora.com","AmerisourceBergen":"cencora.com","JPMorgan Chase":"jpmorganchase.com","Costco Wholesale":"costco.com","Costco":"costco.com","Cigna Group":"thecignagroup.com","Cigna":"cigna.com","Microsoft":"microsoft.com","Cardinal Health":"cardinalhealth.com","Chevron":"chevron.com","Bank of America":"bankofamerica.com","General Motors":"gm.com","Ford Motor":"ford.com","Elevance Health":"elevancehealth.com","Citigroup":"citigroup.com","Meta Platforms":"meta.com","Facebook":"facebook.com","Centene":"centene.com","Home Depot":"homedepot.com","Fannie Mae":"fanniemae.com","Walgreens Boots Alliance":"walgreensbootsalliance.com","Kroger":"kroger.com","Phillips 66":"phillips66.com","Marathon Petroleum":"marathonpetroleum.com","Verizon Communications":"verizon.com","Verizon":"verizon.com","NVIDIA":"nvidia.com","Nvidia":"nvidia.com","Goldman Sachs":"goldmansachs.com","Goldman Sachs Group":"goldmansachs.com","Wells Fargo":"wellsfargo.com","Valero Energy":"valero.com","Comcast":"comcast.com","State Farm Insurance":"statefarm.com","AT&T":"att.com","Freddie Mac":"freddiemac.com","Humana":"humana.com","Morgan Stanley":"morganstanley.com","Target":"target.com","StoneX Group":"stonex.com","Tesla":"tesla.com","Dell Technologies":"delltechnologies.com","PepsiCo":"pepsico.com","Walt Disney":"thewaltdisneycompany.com","Disney":"disney.com","United Parcel Service":"ups.com","UPS":"ups.com","Johnson & Johnson":"jnj.com","FedEx":"fedex.com","Archer Daniels Midland":"adm.com","Procter & Gamble":"pg.com","Lowe's":"lowes.com","Lowes":"lowes.com","Energy Transfer":"energytransfer.com","RTX Corporation":"rtx.com","Raytheon":"rtx.com","Albertsons":"albertsons.com","Sysco":"sysco.com","Progressive":"progressive.com","American Express":"americanexpress.com","Lockheed Martin":"lockheedmartin.com","MetLife":"metlife.com","HCA Healthcare":"hcahealthcare.com","Prudential Financial":"prudential.com","Boeing":"boeing.com","Caterpillar":"caterpillar.com","Merck":"merck.com","Allstate":"allstate.com","Pfizer":"pfizer.com","IBM":"ibm.com","New York Life Insurance":"newyorklife.com","Delta Air Lines":"delta.com","Publix Super Markets":"publix.com","Nationwide":"nationwide.com","TD Synnex":"tdsynnex.com","United Airlines Holdings":"united.com","United Continental Holdings":"united.com","ConocoPhillips":"conocophillips.com","TJX Companies":"tjx.com","TJX":"tjx.com","AbbVie":"abbvie.com","Enterprise Products Partners":"enterpriseproducts.com","Charter Communications":"spectrum.com","Performance Food Group":"pfgc.com","American Airlines Group":"aa.com","Capital One Financial":"capitalone.com","Cisco Systems":"cisco.com","HP Inc":"hp.com","HP":"hp.com","Tyson Foods":"tysonfoods.com","Intel":"intel.com","Oracle":"oracle.com","Broadcom":"broadcom.com","Deere & Company":"deere.com","Deere":"deere.com","Nike":"nike.com","Liberty Mutual Insurance Group":"libertymutual.com","USAA":"usaa.com","Bristol-Myers Squibb":"bms.com","Ingram Micro":"ingrammicro.com","General Dynamics":"gd.com","Coca-Cola":"coca-colacompany.com","TIAA":"tiaa.org","Travelers Companies":"travelers.com","Travelers":"travelers.com","Eli Lilly":"lilly.com","AIG":"aig.com","Dow":"dow.com","Best Buy":"bestbuy.com","Thermo Fisher Scientific":"thermofisher.com","Northrop Grumman":"northropgrumman.com","CHS":"chsinc.com","Abbott Laboratories":"abbott.com","LyondellBasell Industries":"lyondellbasell.com","Qualcomm":"qualcomm.com","Dollar General":"dollargeneral.com","GE Aerospace":"ge.com","General Electric":"ge.com","Salesforce":"salesforce.com","salesforce.com":"salesforce.com","T-Mobile US":"t-mobile.com","Honeywell International":"honeywell.com","Molina Healthcare":"molinahealthcare.com","US Foods Holding":"usfoods.com","Mondelez International":"mondelezinternational.com","PBF Energy":"pbfenergy.com","Northwestern Mutual":"northwesternmutual.com","Philip Morris International":"pmi.com","Nucor":"nucor.com","Jabil":"jabil.com","PACCAR":"paccar.com","MassMutual":"massmutual.com","Cummins":"cummins.com","Amgen":"amgen.com","Medtronic":"medtronic.com","3M":"3m.com","Starbucks":"starbucks.com","Visa":"visa.com","Mastercard":"mastercard.com","BlackRock":"blackrock.com","Netflix":"netflix.com","Adobe":"adobe.com","PayPal Holdings":"paypal.com","eBay":"ebay.com","Intuit":"intuit.com","Advanced Micro Devices":"amd.com","AMD":"amd.com","Robert Half International":"roberthalf.com","Harley-Davidson":"harley-davidson.com","Yum Brands":"yum.com","Levi Strauss":"levi.com"
  };

  var SEED = [
    [1,"Walmart","Retail"],[2,"Amazon","Technology / Retail"],[3,"UnitedHealth Group","Healthcare"],[4,"Apple","Technology"],[5,"CVS Health","Healthcare"],[6,"Berkshire Hathaway","Conglomerate"],[7,"Alphabet","Technology"],[8,"Exxon Mobil","Energy"],[9,"McKesson","Healthcare"],[10,"Cencora","Healthcare"],[11,"JPMorgan Chase","Financial Services"],[12,"Costco Wholesale","Retail"],[13,"Cigna Group","Healthcare"],[14,"Microsoft","Technology"],[15,"Cardinal Health","Healthcare"],[16,"Chevron","Energy"],[17,"Bank of America","Financial Services"],[18,"General Motors","Automotive"],[19,"Ford Motor","Automotive"],[20,"Elevance Health","Healthcare"],[21,"Citigroup","Financial Services"],[22,"Meta Platforms","Technology"],[23,"Centene","Healthcare"],[24,"Home Depot","Retail"],[25,"Fannie Mae","Financial Services"],[26,"Walgreens Boots Alliance","Retail / Pharmacy"],[27,"Kroger","Retail"],[28,"Phillips 66","Energy"],[29,"Marathon Petroleum","Energy"],[30,"Verizon Communications","Telecommunications"],[31,"NVIDIA","Technology"],[32,"Goldman Sachs","Financial Services"],[33,"Wells Fargo","Financial Services"],[34,"Valero Energy","Energy"],[35,"Comcast","Telecommunications / Media"],[36,"State Farm Insurance","Insurance"],[37,"AT&T","Telecommunications"],[38,"Freddie Mac","Financial Services"],[39,"Humana","Healthcare"],[40,"Morgan Stanley","Financial Services"],[41,"Target","Retail"],[42,"StoneX Group","Financial Services"],[43,"Tesla","Automotive / Energy"],[44,"Dell Technologies","Technology"],[45,"PepsiCo","Consumer Goods"],[46,"Walt Disney","Media / Entertainment"],[47,"United Parcel Service","Transportation / Logistics"],[48,"Johnson & Johnson","Pharmaceuticals"],[49,"FedEx","Transportation / Logistics"],[50,"Archer Daniels Midland","Agriculture / Food"],[51,"Procter & Gamble","Consumer Goods"],[52,"Lowe's","Retail"],[53,"Energy Transfer","Energy"],[54,"RTX Corporation","Aerospace / Defense"],[55,"Albertsons","Retail"],[56,"Sysco","Food Distribution"],[57,"Progressive","Insurance"],[58,"American Express","Financial Services"],[59,"Lockheed Martin","Aerospace / Defense"],[60,"MetLife","Insurance"],[61,"HCA Healthcare","Healthcare"],[62,"Prudential Financial","Financial Services"],[63,"Boeing","Aerospace / Defense"],[64,"Caterpillar","Industrial Machinery"],[65,"Merck","Pharmaceuticals"],[66,"Allstate","Insurance"],[67,"Pfizer","Pharmaceuticals"],[68,"IBM","Technology"],[69,"New York Life Insurance","Insurance"],[70,"Delta Air Lines","Airlines"],[71,"Publix Super Markets","Retail"],[72,"Nationwide","Insurance"],[73,"TD Synnex","Technology Distribution"],[74,"United Airlines Holdings","Airlines"],[75,"ConocoPhillips","Energy"],[76,"TJX Companies","Retail"],[77,"AbbVie","Pharmaceuticals"],[78,"Enterprise Products Partners","Energy"],[79,"Charter Communications","Telecommunications"],[80,"Performance Food Group","Food Distribution"],[81,"American Airlines Group","Airlines"],[82,"Capital One Financial","Financial Services"],[83,"Cisco Systems","Technology"],[84,"HP Inc","Technology"],[85,"Tyson Foods","Food / Agriculture"],[86,"Intel","Technology"],[87,"Oracle","Technology"],[88,"Broadcom","Technology"],[89,"Deere & Company","Industrial Machinery"],[90,"Nike","Consumer Goods"],[91,"Liberty Mutual Insurance Group","Insurance"],[92,"Plains GP Holdings","Energy"],[93,"USAA","Insurance"],[94,"Bristol-Myers Squibb","Pharmaceuticals"],[95,"Ingram Micro","Technology Distribution"],[96,"General Dynamics","Aerospace / Defense"],[97,"Coca-Cola","Consumer Goods"],[98,"TIAA","Financial Services"],[99,"Travelers Companies","Insurance"],[100,"Eli Lilly","Pharmaceuticals"],[101,"AIG","Insurance"],[102,"Dow","Chemicals"],[103,"Best Buy","Retail"],[104,"Thermo Fisher Scientific","Life Sciences"],[105,"Northrop Grumman","Aerospace / Defense"],[106,"CHS","Agriculture"],[107,"Abbott Laboratories","Healthcare"],[108,"LyondellBasell Industries","Chemicals"],[109,"Qualcomm","Technology"],[110,"Dollar General","Retail"],[111,"GE Aerospace","Aerospace / Defense"],[112,"Salesforce","Technology"],[113,"T-Mobile US","Telecommunications"],[114,"Honeywell International","Industrial Conglomerate"],[115,"Molina Healthcare","Healthcare"],[116,"US Foods Holding","Food Distribution"],[117,"Mondelez International","Consumer Goods"],[118,"PBF Energy","Energy"],[119,"Northwestern Mutual","Insurance"],[120,"Philip Morris International","Tobacco"],[121,"Nucor","Metals / Steel"],[122,"Jabil","Technology Manufacturing"],[123,"PACCAR","Automotive / Trucks"],[124,"MassMutual","Insurance"],[125,"Cummins","Industrial Machinery"],[126,"Amgen","Pharmaceuticals / Biotech"],[127,"Medtronic","Healthcare / Medical Devices"],[128,"3M","Industrial Conglomerate"]
  ];

  var companies = [];
  var bySlug = {};
  var readyPromise = load();

  function slugify(s) { return String(s || "").toLowerCase().replace(/&/g," and ").replace(/[^a-z0-9]+/g,"-").replace(/^-+|-+$/g,""); }
  function esc(s) { return String(s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }
  function initials(name) { return String(name || "?").replace(/&/g," ").split(/\s+/).filter(Boolean).slice(0,2).map(function(w){return w[0];}).join("").toUpperCase(); }
  function domainGuess(name) { return String(name || "").toLowerCase().replace(/&/g,"and").replace(/\b(company|companies|corporation|corp|inc|holdings|group|international|industries|systems|technologies|technology|financial|services|insurance|the)\b/g,"").replace(/[^a-z0-9]/g,"") + ".com"; }
  function normalizeRecord(r, i) {
    var name = r.company || r.name || r[1] || "Company " + (i + 1);
    var rank = Number(r.rank || r[0] || i + 1);
    var industry = r.industry || r[2] || inferIndustry(name);
    var slug = slugify(name);
    var domain = r.domain || DOMAIN_OVERRIDES[name] || domainGuess(name);
    return { rank: rank, name: name, company: name, industry: industry, revenueMillions: r.revenueMillions || r.revenue || null, employees: r.employees || null, headquarters: r.headquarters || "", state: r.state || "", slug: slug, domain: domain };
  }
  function inferIndustry(name) {
    var n = String(name).toLowerCase();
    if (/bank|financial|capital|credit|visa|mastercard|amex|express/.test(n)) return "Financial Services";
    if (/health|medical|pharma|lilly|pfizer|abbott|amgen|merck/.test(n)) return "Healthcare";
    if (/energy|oil|petroleum|gas|chevron|exxon|valero/.test(n)) return "Energy";
    if (/air|boeing|aerospace|defense|lockheed|raytheon/.test(n)) return "Aerospace / Defense";
    if (/motor|auto|tesla|ford|gm|deere/.test(n)) return "Automotive / Industrial";
    if (/tech|data|software|systems|oracle|microsoft|apple|nvidia|intel|cisco|adobe/.test(n)) return "Technology";
    if (/insurance|mutual|life|allstate|farm/.test(n)) return "Insurance";
    if (/walmart|costco|target|kroger|dollar|stores|retail|depot|lowe/.test(n)) return "Retail";
    return "Fortune 500";
  }
  function setCompanies(list) {
    companies = list.map(normalizeRecord).filter(function(c){ return c.name; }).slice(0, 500);
    bySlug = {};
    companies.forEach(function(c){ bySlug[c.slug] = c; });
  }
  function parseCsv(text) {
    return String(text || "").split(/\r?\n/).slice(1).map(function(line){
      var m = line.match(/^(\d+),(.+?),(\d+(?:\.\d+)?),/);
      return m ? { rank: Number(m[1]), company: m[2], industry: inferIndustry(m[2]) } : null;
    }).filter(Boolean);
  }
  function load() {
    setCompanies(SEED);
    return fetch(DATA_URLS[0], { cache: "no-store" }).then(function(r){ if(!r.ok) throw new Error("local missing"); return r.text(); })
      .catch(function(){ return fetch(DATA_URLS[1], { cache: "force-cache" }).then(function(r){ if(!r.ok) throw new Error("salt missing"); return r.json(); }); })
      .catch(function(){ return fetch(DATA_URLS[2], { cache: "force-cache" }).then(function(r){ if(!r.ok) throw new Error("github missing"); return r.text(); }); })
      .then(function(payload){
        if (payload && payload.data) setCompanies(payload.data);
        else if (typeof payload === "string") setCompanies(parseCsv(payload));
        return companies;
      })
      .catch(function(){ return companies; });
  }
  function logoUrl(c, size) {
    var domain = (c && c.domain) || domainGuess(c && c.name);
    return "https://www.google.com/s2/favicons?domain=" + encodeURIComponent(domain) + "&sz=" + ((size || 64) * 2);
  }
  function fallbackSvg(c, size) {
    var label = initials(c && c.name);
    var svg = '<svg xmlns="http://www.w3.org/2000/svg" width="'+size+'" height="'+size+'" viewBox="0 0 '+size+' '+size+'"><rect width="100%" height="100%" rx="12" fill="#fff7ed"/><text x="50%" y="54%" text-anchor="middle" dominant-baseline="middle" font-family="Arial, sans-serif" font-size="'+Math.max(12, size/3)+'" font-weight="800" fill="#7c2d12">'+esc(label)+'</text></svg>';
    return "data:image/svg+xml;charset=UTF-8," + encodeURIComponent(svg);
  }
  function logoImgHtml(c, size, className) {
    size = size || 40;
    return '<img class="fortune-logo-img '+esc(className || '')+'" src="'+esc(logoUrl(c, size))+'" width="'+size+'" height="'+size+'" alt="'+esc((c && c.name) || "Company")+' logo" loading="lazy" decoding="async" onerror="this.onerror=null;this.src=\''+fallbackSvg(c, size).replace(/'/g, "%27")+'\';" />';
  }

  window.Fortune500 = {
    ready: function(){ return readyPromise; },
    list: function(){ return companies.slice(); },
    getBySlug: function(slug){ return bySlug[slug] || null; },
    searchCompanies: function(q, limit){
      q = String(q || "").toLowerCase().trim();
      if (!q) return [];
      return companies.filter(function(c){ return c.name.toLowerCase().includes(q) || c.slug.includes(q) || String(c.industry || "").toLowerCase().includes(q); }).slice(0, limit || 20);
    },
    logoUrl: logoUrl,
    logoImgHtml: logoImgHtml,
    iconMarkSvg: function(c, size){ return logoImgHtml(c, size || 40, ""); },
    fallbackSvg: fallbackSvg
  };
})();
