(function () {
  'use strict';

  angular
    .module('platform.posts.services')
    .factory('Posts', Posts);

  Posts.$inject = ['$http'];


  function Posts($http) {
    var Posts = {
      all: all,
      create: create,
      get: get
    };

    return Posts;


    function all() {
      return $http.get('/api/v1/posts/');
    }


    function create(fd) {
      return $http.post('/api/v1/posts/', fd, {
        withCredentials: true,
        headers: {'Content-Type': undefined },
        transformRequest: angular.identity
      }).then(createSuccessFn, createErrorFn);

      function createSuccessFn(data, status, headers, config) {
        console.log('Post created with succcess.');
        window.location = '/';
      }

      function createErrorFn(data, status, headers, config) {
        console.error('Could not create post.');
      }
    }


    function get(username) {
      return $http.get('/api/v1/accounts/' + username + '/posts/');
    }
  }
})();
