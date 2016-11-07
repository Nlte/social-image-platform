/**
* Posts
* @namespace platform.posts.services
*/
(function () {
  'use strict';

  angular
    .module('platform.posts.services')
    .factory('Posts', Posts);

  Posts.$inject = ['$http'];

  /**
  * @namespace Posts
  * @returns {Factory}
  */
  function Posts($http) {
    var Posts = {
      all: all,
      create: create,
      get: get
    };

    return Posts;

    ////////////////////

    /**
    * @name all
    * @desc Get all Posts
    * @returns {Promise}
    * @memberOf platform.posts.services.Posts
    */
    function all() {
      return $http.get('/api/v1/posts/');
    }


    /**
    * @name create
    * @desc Create a new Post
    * @param {string} user_caption The user_caption of the new Post
    * @returns {Promise}
    * @memberOf platform.posts.services.Posts
    */
    function create(user_caption) {
      return $http.post('/api/v1/posts/', {
        user_caption: user_caption
      });
    }

    /**
     * @name get
     * @desc Get the Posts of a given user
     * @param {string} username The username to get Posts for
     * @returns {Promise}
     * @memberOf platform.posts.services.Posts
     */
    function get(username) {
      return $http.get('/api/v1/accounts/' + username + '/posts/');
    }
  }
})();
