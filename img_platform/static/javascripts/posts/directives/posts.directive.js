(function () {
  'use strict';

  angular
    .module('platform.posts.directives')
    .directive('posts', posts);


  function posts() {

    var directive = {
      controller: 'PostsController',
      controllerAs: 'vm',
      restrict: 'E',
      scope: {
        posts: '='
      },
      templateUrl: '/static/templates/posts/posts.html'
    };

    return directive;
  }
})();
