angular
  .module('platform', [
    'platform.config',
    'platform.routes',
    'platform.authentication',
    'platform.layout',
    'platform.posts',
    'platform.utils',
    'platform.profiles'
  ]);

angular
  .module('platform.config', []);

angular
  .module('platform.routes', ['ngRoute']);

angular
  .module('platform')
  .run(run);

run.$inject = ['$http'];

/**
* @name run
* @desc Update xsrf $http headers to align with Django's defaults
*/
function run($http) {
  $http.defaults.xsrfHeaderName = 'X-CSRFToken';
  $http.defaults.xsrfCookieName = 'csrftoken';
}
