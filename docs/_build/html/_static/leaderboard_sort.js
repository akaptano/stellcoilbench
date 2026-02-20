/**
 * Leaderboard table sorting - attaches click handlers to sortable column headers.
 * Works with any table that has class "leaderboard-sortable" and th.sortable headers.
 */
(function() {
  function initSortableTables() {
    var tables = document.querySelectorAll('table.leaderboard-sortable');
    tables.forEach(function(table) {
      var headers = table.querySelectorAll('th.sortable');
      if (headers.length === 0) return;
      var sortDir = {};
      headers.forEach(function(th, i) { sortDir[i] = 1; });
      function sortTable(col) {
        var tbody = table.querySelector('tbody');
        if (!tbody) return;
        var rows = Array.from(tbody.querySelectorAll('tr'));
        sortDir[col] = sortDir[col] || 1;
        var mult = sortDir[col];
        sortDir[col] = -sortDir[col];
        rows.sort(function(a, b) {
          var ac = a.children[col];
          var bc = b.children[col];
          var av = ac && ac.getAttribute('data-sort-value');
          var bv = bc && bc.getAttribute('data-sort-value');
          var an = parseFloat(av);
          var bn = parseFloat(bv);
          if (!isNaN(an) && !isNaN(bn)) return mult * (an - bn);
          return mult * String(av || '').localeCompare(String(bv || ''));
        });
        rows.forEach(function(r) { tbody.appendChild(r); });
      }
      headers.forEach(function(th, i) {
        th.addEventListener('click', function() { sortTable(i); });
      });
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSortableTables);
  } else {
    initSortableTables();
  }
})();
