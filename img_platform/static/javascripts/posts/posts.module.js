(function () {
  'use strict';

  angular
    .module('platform.posts', [
      'platform.posts.controllers',
      'platform.posts.directives',
      'platform.posts.services'
    ]);

  angular
    .module('platform.posts.controllers', []);

  angular
    .module('platform.posts.directives', ['ngDialog']);

  angular
    .module('platform.posts.services', []);
})();